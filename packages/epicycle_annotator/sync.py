"""Label synchronization between main viewer and native subviewers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import numpy as np
from napari.layers import Labels


# ---------------------------------------------------------------------------
# Brush-size lock (2026-05-17)
# ---------------------------------------------------------------------------
#
# Locked design decision: brush_size is fixed at 1 on every
# annotator Labels layer (main, subviewers, isolation proxy, multi-label
# sub-proxies, refine layer).  The slider in napari's QtLabelsControls
# remains visible — if the user drags it, the snap-back handler reverts
# the value to 1.  Adjustable brush is a future feature; it must be
# designed alongside a proper sync redesign that handles the
# main / proxy / subviewer disagreement (see history before 2026-05-17 for
# the wiring bugs this lock works around).
#
# Perf contract: the snap-back handler is a strict no-op when brush_size
# is already 1.  Reentrancy is impossible because we check value-before-
# write and use a per-layer guard.  Steady state cost = 1 integer compare
# per brush_size event, which napari emits only on actual change.
#
# Unlock path (when adjustable brush ships): delete the
# `lock_brush_size_to_1` calls at the layer creation sites, remove the
# `brush_size = 1` hardcoded line in `sync_label_properties`, and remove
# the analogous hardcoded line in `_apply_draw_settings`.  The sync
# manager's brush_size event handlers can stay — they will resume
# propagating user values.

_BRUSH_LOCKED_VALUE = 1


def lock_brush_size_to_1(layer) -> None:
    """Lock a Labels layer's ``brush_size`` to 1.

    Sets ``brush_size = 1`` immediately, then installs a snap-back event
    handler that reverts any subsequent change.  Idempotent — calling
    twice on the same layer attaches the handler only once.

    Why this exists: see the module-level comment above.  Briefly,
    pre-2026-05-17 the sync wiring between main / isolation proxy /
    subviewers had three bugs that produced disagreement on brush_size.
    Rather than patch each bug in isolation, brush_size is fixed at 1
    until adjustable brush is shipped as a designed feature.

    Perf: the snap-back handler is a strict no-op when ``brush_size``
    is already 1 (it checks value-before-write to avoid emitting the
    event again).  Steady state cost is one integer compare per actual
    brush_size change event, which napari emits only on real change.

    Parameters
    ----------
    layer
        A napari Labels layer (or any object with ``brush_size`` and
        ``events.brush_size``).  Tolerant of fakes — wrapped in try/except
        so tests using ``MagicMock`` don't crash.
    """
    # Seed: write 1 first so the layer starts locked even before the
    # handler is attached.  Setting to the locked value is safe-no-op
    # if napari is already there.
    try:
        if int(getattr(layer, "brush_size", _BRUSH_LOCKED_VALUE)) != _BRUSH_LOCKED_VALUE:
            layer.brush_size = _BRUSH_LOCKED_VALUE
    except Exception:
        pass

    # Idempotency guard: don't attach twice.
    if getattr(layer, "_epicycle_brush_locked", False):
        return
    try:
        layer._epicycle_brush_locked = True
    except Exception:
        return

    # Attach snap-back handler.  Defensive in case the fake/test layer
    # doesn't expose events.brush_size.
    try:
        emitter = getattr(getattr(layer, "events", None), "brush_size", None)
    except Exception:
        emitter = None
    if emitter is None:
        return

    def _snap_back(event):
        src = getattr(event, "source", layer)
        try:
            current = int(getattr(src, "brush_size", _BRUSH_LOCKED_VALUE))
        except Exception:
            return
        if current == _BRUSH_LOCKED_VALUE:
            return  # already locked — no work, no event re-emit
        # Reentrancy guard: prevents a hypothetical edge case where
        # writing _BRUSH_LOCKED_VALUE fires the event before the value
        # is actually read.  Belt-and-suspenders alongside the
        # value-check above.
        if getattr(src, "_epicycle_brush_snapping", False):
            return
        try:
            src._epicycle_brush_snapping = True
            src.brush_size = _BRUSH_LOCKED_VALUE
            print(
                f"[BRUSH-LOCK] snapped back layer={getattr(src, 'name', '?')!r} "
                f"{current} -> {_BRUSH_LOCKED_VALUE}"
            )
        finally:
            try:
                src._epicycle_brush_snapping = False
            except Exception:
                pass

    try:
        emitter.connect(_snap_back)
        # Stash the callback so cleanup paths (if any future code adds
        # a deregister step) can find it.
        layer._epicycle_brush_snap_cb = _snap_back
    except Exception:
        pass


class LabelSyncManager:
    """Manage shared label arrays across main viewer and orthogonal subviewers."""

    def __init__(
        self,
        main_viewer,
        viewer_model_xy,
        viewer_model_xz,
        viewer_model_yz,
        qt_viewer_xy,
        qt_viewer_xz,
        qt_viewer_yz,
    ) -> None:
        self.main_viewer = main_viewer
        self.viewer_model_xy = viewer_model_xy
        self.viewer_model_xz = viewer_model_xz
        self.viewer_model_yz = viewer_model_yz
        self.qt_viewer_xy = qt_viewer_xy
        self.qt_viewer_xz = qt_viewer_xz
        self.qt_viewer_yz = qt_viewer_yz

        self.shared_labels: Dict[str, np.ndarray] = {}
        self.edited_slices: Dict[str, Dict[str, Dict[int, list[int]]]] = {}

        # Handler references for clean disconnection on close/relaunch.
        # Keyed by id(label_layer) → list of (event_emitter, handler).
        self._sync_handlers: Dict[int, list] = {}
        # Keyed by id(label_layer) → (drag_callback, n_edit_handler_or_None).
        self._main_drag_handlers: Dict[int, tuple] = {}
        # Nesting counter for suppress_property_sync context manager.
        self._suppress_depth: int = 0
        # When True, sync_selected_label skips show_selected_label propagation.
        # Set by the 3D isolation proxy to prevent the main layer's
        # show_selected_label=False from stomping subviewer isolation state.
        self._suppress_show_selected_sync: bool = False

        # Cached "next free label" counter for enclosed-fill auto-increment
        # (2026-05-28, fix/step2-autoinc-perf).  Pre-fix, every enclosed-fill
        # ran np.max over ALL work_labels volumes (~265ms on busy axial
        # patches — 4 full-volume scans).  Now we seed this counter ONCE via
        # a scan, then increment it per fill (instant).  None = needs
        # (re)seeding; invalidated whenever labels are introduced outside
        # the fill path (see invalidate_next_fill_label_cache).
        self._next_fill_label_cache = None

    @contextmanager
    def suppress_property_sync(self):
        """Context manager to suppress sync_label_properties during batch
        colormap operations.

        Uses a nesting counter so nested contexts are safe.  While
        suppressed, ``sync_label_properties`` returns immediately,
        preventing the N × 3-subviewer GPU cascade for each
        ``layer.colormap = ...`` assignment inside the batch.
        """
        self._suppress_depth += 1
        self._syncing_properties = True
        try:
            yield
        finally:
            self._suppress_depth -= 1
            if self._suppress_depth <= 0:
                self._suppress_depth = 0
                self._syncing_properties = False

    def _base_label_name(self, name: str) -> str:
        text = str(name or "")
        for suffix in ("_XY", "_XZ", "_YZ"):
            if text.endswith(suffix):
                return text[: -len(suffix)]
        return text

    def _layer_by_name(self, layer_list, name: str):
        try:
            if name in layer_list:
                return layer_list[name]
        except Exception:
            pass
        for layer in layer_list:
            if getattr(layer, "name", "") == name:
                return layer
        return None

    def register_shared_data(self, layer_name: str, data: np.ndarray) -> None:
        self.shared_labels[layer_name] = data

    def unregister_layer(self, layer_name: str) -> None:
        self.shared_labels.pop(layer_name, None)
        self.edited_slices.pop(layer_name, None)

    def get_edited_slices(self, layer_name: str) -> dict:
        """Return edited slices dict for a layer.

        Returns ``{"XY": {label_id: [indices]}, "XZ": {...}, "YZ": {...}}``.
        """
        return self.edited_slices.get(layer_name, {"XY": {}, "XZ": {}, "YZ": {}})

    def clear_edited_slices_for_label(self, layer_name: str, label_id: int) -> None:
        """Remove all edited-slice entries for *label_id* across orientations.

        Handles both native-int and numpy-int keys.
        """
        edited = self.edited_slices.get(layer_name)
        if edited is None:
            return
        for orient_dict in edited.values():
            orient_dict.pop(label_id, None)
            for key in list(orient_dict.keys()):
                if int(key) == label_id:
                    orient_dict.pop(key, None)

    def rename_layer(self, old_name: str, new_name: str) -> None:
        if old_name in self.shared_labels:
            self.shared_labels[new_name] = self.shared_labels.pop(old_name)
        if old_name in self.edited_slices:
            self.edited_slices[new_name] = self.edited_slices.pop(old_name)

    def connect_label_sync(self, label_layer: Labels) -> None:
        # Guard: only connect once per layer to avoid handler accumulation.
        if getattr(label_layer, "_epicycle_sync_connected", False):
            return
        label_layer._epicycle_sync_connected = True

        handlers = []  # (event_emitter, handler) for clean disconnect

        mode_cb = lambda _e: self.sync_label_modes()
        label_layer.events.mode.connect(mode_cb)
        handlers.append((label_layer.events.mode, mode_cb))

        sel_cb = lambda e: self.sync_selected_label(getattr(e, "source", label_layer))
        label_layer.events.selected_label.connect(sel_cb)
        handlers.append((label_layer.events.selected_label, sel_cb))

        brush_cb = lambda e: self.sync_label_properties(getattr(e, "source", label_layer))
        label_layer.events.brush_size.connect(brush_cb)
        handlers.append((label_layer.events.brush_size, brush_cb))

        opacity_cb = lambda e: self.sync_label_properties(getattr(e, "source", label_layer))
        label_layer.events.opacity.connect(opacity_cb)
        handlers.append((label_layer.events.opacity, opacity_cb))

        cmap_cb = lambda e: self.sync_label_properties(getattr(e, "source", label_layer))
        label_layer.events.colormap.connect(cmap_cb)
        handlers.append((label_layer.events.colormap, cmap_cb))

        blend_cb = lambda e: self.sync_label_properties(getattr(e, "source", label_layer))
        label_layer.events.blending.connect(blend_cb)
        handlers.append((label_layer.events.blending, blend_cb))

        # Mirror contour (outline) across main + subviewers, so bumping napari's
        # built-in contour control on the main labels layer shows the outline in
        # the XY/XZ/YZ side views too (helps eyeball cell merges).  Same path as
        # opacity/blending above; defensive hasattr mirrors the n_edit pattern.
        if hasattr(label_layer.events, "contour"):
            contour_cb = lambda e: self.sync_label_properties(
                getattr(e, "source", label_layer)
            )
            label_layer.events.contour.connect(contour_cb)
            handlers.append((label_layer.events.contour, contour_cb))

        if hasattr(label_layer.events, "n_edit_dimensions"):
            nedit_cb = lambda e: self.sync_label_properties(getattr(e, "source", label_layer))
            label_layer.events.n_edit_dimensions.connect(nedit_cb)
            handlers.append((label_layer.events.n_edit_dimensions, nedit_cb))

        # ROOT-CAUSE FIX for shared_data ↔ layer.data divergence:
        # napari's Labels.data setter fires events.data when reassigned
        # (e.g., controller paths like _zero_label_ids do
        #   `data = layer.data.copy(); data[mask] = 0; layer.data = data`).
        # After such a reassignment, sync_manager.shared_labels[name]
        # points at the OLD numpy array, while layer.data is the NEW one.
        # Refinement ops that read shared_data (split, smooth, interp,
        # merge) then see stale state — observable as truncated snapshots
        # / clean horizontal slice cuts in the volume after reject.
        # Listener is a dict reference update only — no data copy, O(1).
        # Defensive `hasattr` guard mirrors the n_edit_dimensions pattern
        # above — fakes in tests don't expose events.data.
        if hasattr(label_layer.events, "data"):
            def _data_reassigned_cb(_e):
                try:
                    cur_name = label_layer.name
                    self.shared_labels[cur_name] = label_layer.data
                except Exception:
                    pass
            label_layer.events.data.connect(_data_reassigned_cb)
            handlers.append((label_layer.events.data, _data_reassigned_cb))

        self._sync_handlers[id(label_layer)] = handlers

    def disconnect_label_sync(self, label_layer: Labels) -> None:
        """Disconnect all event handlers added by connect_label_sync.

        Removes stored handlers and resets the guard flag so a new
        SyncManager can reconnect on relaunch.
        """
        layer_id = id(label_layer)
        handlers = self._sync_handlers.pop(layer_id, [])
        for event_emitter, handler in handlers:
            try:
                event_emitter.disconnect(handler)
            except (ValueError, AttributeError):
                pass
        label_layer._epicycle_sync_connected = False

    def sync_label_properties(self, active_layer: Labels) -> None:
        # Reentrant guard — setting properties on subviewers fires their
        # events which call sync again.  Without this guard each property
        # change cascades N² times (main + 3 subs × 3 subs).
        if getattr(self, "_syncing_properties", False):
            return
        self._syncing_properties = True
        try:
            # brush_size is locked at 1 across all annotator layers — see
            # lock_brush_size_to_1 docstring.  Hardcoding here means even if the lock
            # handler hasn't snapped back yet, the propagated value is 1.
            brush_size = _BRUSH_LOCKED_VALUE
            opacity = float(getattr(active_layer, "opacity", 1.0))
            colormap = getattr(active_layer, "colormap", None)
            blending = getattr(active_layer, "blending", "translucent")
            n_edit_dim = int(getattr(active_layer, "n_edit_dimensions", 2))
            contour = int(getattr(active_layer, "contour", 0))
            base = self._base_label_name(str(getattr(active_layer, "name", "")))
            # Also propagate to main layer (covers subviewer → main direction).
            main_layer = self._layer_by_name(self.main_viewer.layers, base)
            if isinstance(main_layer, Labels) and main_layer is not active_layer:
                main_layer.brush_size = brush_size
                main_layer.opacity = opacity
                if colormap is not None:
                    main_layer.colormap = colormap
                main_layer.blending = blending
                try:
                    main_layer.n_edit_dimensions = n_edit_dim
                except Exception:
                    pass
                try:
                    main_layer.contour = contour
                except Exception:
                    pass
            for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
                for layer in vm.layers:
                    if isinstance(layer, Labels) and self._base_label_name(layer.name) == base:
                        layer.brush_size = brush_size
                        layer.opacity = opacity
                        if colormap is not None:
                            layer.colormap = colormap
                        layer.blending = blending
                        try:
                            layer.n_edit_dimensions = n_edit_dim
                        except Exception:
                            pass
                        try:
                            layer.contour = contour
                        except Exception:
                            pass
        finally:
            self._syncing_properties = False

    def _scan_global_max_label(self, active_layer) -> int:
        """One-time scan of the global max label across ``active_layer`` +
        all main-viewer Labels layers.  SEEDS the next-fill-label cache.

        This is the expensive op (np.max over every full work_labels
        volume) that used to run on EVERY enclosed-fill (~265ms on busy
        axial patches).  Now it runs only when the cache needs (re)seeding.
        """
        vol_max = 0
        try:
            vol_max = int(np.max(active_layer.data))
        except Exception:
            vol_max = 0
        try:
            for lyr in self.main_viewer.layers:
                if isinstance(lyr, Labels) and lyr is not active_layer:
                    try:
                        vol_max = max(vol_max, int(np.max(lyr.data)))
                    except Exception:
                        pass
        except Exception:
            pass
        return vol_max

    def next_fill_label(self, active_layer, just_painted: int = 0) -> int:
        """Return the next unique label ID for an enclosed-fill, WITHOUT the
        per-fill full-volume scan (2026-05-28, fix/step2-autoinc-perf).

        Lazily seeds a cached running counter on first use (one scan via
        ``_scan_global_max_label``), then increments it per fill (instant).

        ``just_painted`` is the label the fill just used (the pre-bump
        ``selected_label``).  We take ``max(cache, just_painted)`` before
        incrementing so a manually-overridden or R-picked label that's
        higher than the counter still advances it — preventing the next
        fill from colliding with it.

        Correctness: the counter can never go stale-LOW (→ colliding ID →
        merged cells) because every OTHER label-introducing path
        (predictions / restore / segment, all funnelling through the
        controller's ``_insert_slice_mask_with_unique_labels``) calls
        ``invalidate_next_fill_label_cache`` to force a fresh re-seed.
        """
        if self._next_fill_label_cache is None:
            self._next_fill_label_cache = self._scan_global_max_label(active_layer)
        try:
            jp = int(just_painted)
        except (TypeError, ValueError):
            jp = 0
        self._next_fill_label_cache = max(self._next_fill_label_cache, jp)
        self._next_fill_label_cache += 1
        return max(1, self._next_fill_label_cache)

    def invalidate_next_fill_label_cache(self) -> None:
        """Drop the cached next-fill-label counter so the next enclosed-fill
        re-seeds it with a fresh global-max scan.

        MUST be called after any path that introduces labels OUTSIDE
        ``next_fill_label`` (predictions, restore, segment) — otherwise the
        counter could hand out an ID that collides with a just-inserted
        label, merging two cells.  Wired into the controller's
        ``_insert_slice_mask_with_unique_labels`` chokepoint (the single
        funnel for all label-introducing inserts)."""
        self._next_fill_label_cache = None

    def sync_selected_label(self, source_layer: Labels) -> None:
        # Reentrant guard — same cascade prevention as sync_label_properties.
        if getattr(self, "_syncing_selected_label", False):
            return
        self._syncing_selected_label = True
        try:
            selected = int(getattr(source_layer, "selected_label", 1))
            # When the 3D isolation proxy is active, the main layer
            # deliberately has show_selected_label=False to avoid expensive
            # iso_categorical re-render.  Propagating that False would stomp
            # the subviewers' show_selected_label=True (which the annotator
            # set explicitly for 2D isolation display).  Skip show_sel
            # propagation entirely when suppressed — the annotator manages
            # subviewer show_selected_label directly via
            # _propagate_show_selected_label().
            _suppress_show_sel = getattr(self, "_suppress_show_selected_sync", False)
            show_sel = (
                None if _suppress_show_sel
                else getattr(source_layer, "show_selected_label", None)
            )
            base = self._base_label_name(str(getattr(source_layer, "name", "")))
            main_layer = self._layer_by_name(self.main_viewer.layers, base)
            if isinstance(main_layer, Labels):
                main_layer.selected_label = selected
                # Sync show_selected_label atomically with selected_label
                # to prevent desync between main viewer and subviewers.
                if show_sel is not None and hasattr(main_layer, "show_selected_label"):
                    main_layer.show_selected_label = show_sel
            for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
                for layer in vm.layers:
                    if isinstance(layer, Labels) and self._base_label_name(layer.name) == base:
                        layer.selected_label = selected
                        if show_sel is not None and hasattr(layer, "show_selected_label"):
                            layer.show_selected_label = show_sel
        finally:
            self._syncing_selected_label = False

    def sync_label_modes(self) -> None:
        main_layer = self.main_viewer.layers.selection.active
        if not isinstance(main_layer, Labels):
            return
        for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
            for layer in vm.layers:
                if isinstance(layer, Labels) and layer.name.startswith(main_layer.name):
                    layer.mode = main_layer.mode

    def sync_active_layer(self, _event=None) -> None:
        active_layer = self.main_viewer.layers.selection.active
        if active_layer is None:
            return
        name = active_layer.name
        for orientation, vm, qt in zip(
            ["XY", "XZ", "YZ"],
            [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz],
            [self.qt_viewer_xy, self.qt_viewer_xz, self.qt_viewer_yz],
        ):
            target = self._layer_by_name(vm.layers, f"{name}_{orientation}")
            if target is None:
                target = self._layer_by_name(vm.layers, name)
            if target is None:
                continue
            vm.layers.selection.active = target
            qt.viewer.layers.selection.active = target
            if hasattr(active_layer, "mode") and hasattr(target, "mode"):
                target.mode = active_layer.mode

    def track_slice_edit(self, layer: Labels, viewer_model, orientation: str) -> None:
        """Record that the user edited *layer* on the current slice.

        Only the ``selected_label`` is tracked (the label the user is
        actively painting with).  Previous versions scanned ``np.unique``
        over the entire slice which was extremely slow on dense label
        volumes and caused every pre-existing label to be registered as
        "edited", leading to runaway interpolation.
        """
        orient = str(orientation).upper()
        layer_base_name = layer.name.replace(f"_{orient}", "")
        if layer_base_name not in self.edited_slices:
            self.edited_slices[layer_base_name] = {"XY": {}, "XZ": {}, "YZ": {}}

        slice_index = int(viewer_model.dims.current_step[0])
        orient_dict = self.edited_slices[layer_base_name][orient]

        # Only track the label the user is currently painting/erasing.
        selected = int(getattr(layer, "selected_label", 0))
        if selected > 0:
            orient_dict.setdefault(selected, [])
            if slice_index not in orient_dict[selected]:
                orient_dict[selected].append(slice_index)

    def sync_to_main_view(
        self, layer: Labels, orientation: str, viewer_model,
        *, defer_main_refresh: bool = False,
    ) -> None:
        orient = str(orientation).upper()
        layer_base_name = layer.name.replace(f"_{orient}", "")
        if layer_base_name not in self.shared_labels:
            return

        shared_data = self.shared_labels[layer_base_name]
        if orient == "XY":
            idx = int(viewer_model.dims.current_step[0])
            shared_data[idx, :, :] = layer.data[idx, :, :]
        elif orient == "XZ":
            idx = int(viewer_model.dims.current_step[0])
            shared_data[:, idx, :] = layer.data[idx, :, :]
        elif orient == "YZ":
            # YZ subviewer data shape is (X, Y, Z) via transpose(2,1,0).
            # current_step[0] = X index (the slice being viewed).
            # layer.data[idx, :, :] gives (Y, Z) slice.
            # shared_data[:, :, idx] gives (Z, Y) slice → need .T.
            idx = int(viewer_model.dims.current_step[0])
            shared_data[:, :, idx] = layer.data[idx, :, :].T
        else:
            return

        main_layer = self._layer_by_name(self.main_viewer.layers, layer_base_name)
        if main_layer is not None:
            # If dtype normalization created a separate array, copy only
            # the modified slice instead of the full volume.
            if main_layer.data is not shared_data:
                if orient == "XY":
                    main_layer.data[idx, :, :] = shared_data[idx, :, :]
                elif orient == "XZ":
                    main_layer.data[:, idx, :] = shared_data[:, idx, :]
                elif orient == "YZ":
                    main_layer.data[:, :, idx] = shared_data[:, :, idx]
            if not defer_main_refresh:
                main_layer.refresh()

    def connect_main_label_sync(self, label_layer: Labels) -> None:
        """Add mouse_drag callback + brush-size + n-edit sync on a main-
        viewer Labels layer.

        Pre-2026-05-10: only ``n_edit_dimensions`` was connected on the
        main layer.  Setting ``brush_size`` on the main slider fired
        napari's events but no listener was attached → no propagation
        to subviewers.

        Fix (D-brush-sync, 2026-05-10, narrowed): connect ``brush_size``
        in addition to ``n_edit_dimensions``.

        DELIBERATELY NOT connected: ``colormap``, ``opacity``, ``blending``.
        Reasons:
          1. ``colormap`` on the main layer is reassigned by isolation
             (``_apply_multi_isolation_colormap``), QC coloring
             (``_update_viewer_feature_colors``), and proxy create/teardown.
             Each of these ALREADY explicitly propagates to subviewers
             via ``_propagate_colormap_to_subviewers``.  A duplicate
             event-driven path runs the propagation a second time on
             every isolation transition and — critically — can fire
             during patch teardown when the source main layer or
             sibling subviewers are mid-removal, causing
             sip-deleted-layer access → 3D crash.  (Original observation
             2026-05-10: 3D crashes started immediately after the
             original 4-event hook landed; reverting to brush-only
             eliminated them.)
          2. ``opacity`` and ``blending`` follow the same risk profile
             and are not user-changed often enough to justify the
             attack surface.
        """
        if getattr(label_layer, "_epicycle_main_drag_connected", False):
            return
        label_layer._epicycle_main_drag_connected = True
        callback = self._create_main_drag_callback(label_layer)
        label_layer.mouse_drag_callbacks.append(callback)
        prop_cbs = []
        if hasattr(label_layer, "events"):
            sync_props_cb = lambda e: self.sync_label_properties(
                getattr(e, "source", label_layer)
            )
            brush_emitter = getattr(label_layer.events, "brush_size", None)
            if brush_emitter is not None:
                brush_emitter.connect(sync_props_cb)
                prop_cbs.append((brush_emitter, sync_props_cb))
            n_edit_emitter = getattr(label_layer.events, "n_edit_dimensions", None)
            if n_edit_emitter is not None:
                n_edit_emitter.connect(sync_props_cb)
                prop_cbs.append((n_edit_emitter, sync_props_cb))
        self._main_drag_handlers[id(label_layer)] = (callback, prop_cbs)

    def disconnect_main_label_sync(self, label_layer: Labels) -> None:
        """Disconnect handlers added by connect_main_label_sync.

        Removes the drag callback and all property-event handlers
        (brush_size, opacity, colormap, blending, n_edit_dimensions),
        then resets the guard flag so a new SyncManager can reconnect.
        """
        layer_id = id(label_layer)
        entry = self._main_drag_handlers.pop(layer_id, None)
        if entry is not None:
            drag_cb, prop_cbs = entry
            try:
                label_layer.mouse_drag_callbacks.remove(drag_cb)
            except ValueError:
                pass
            # prop_cbs is a list of (emitter, callback) tuples.  Older
            # entries (pre-D-brush-sync 2026-05-10) stored a single
            # n_edit_cb or None — handle both shapes for backwards-compat
            # if the dict was populated mid-session by an older code path.
            if isinstance(prop_cbs, list):
                for emitter, callback in prop_cbs:
                    try:
                        emitter.disconnect(callback)
                    except (ValueError, AttributeError):
                        pass
            elif prop_cbs is not None:
                # Backwards-compat: single n_edit_cb shape from older code.
                try:
                    label_layer.events.n_edit_dimensions.disconnect(prop_cbs)
                except (ValueError, AttributeError):
                    pass
        label_layer._epicycle_main_drag_connected = False

    def disconnect_all(self) -> None:
        """Flush ALL tracked handler dicts.

        Called as a belt-and-suspenders after per-layer disconnect loops
        to ensure no stale entries remain (e.g. for layers that were garbage
        collected between connect and disconnect).

        For each remaining entry, disconnects event-emitter-based handlers
        where possible. Dict-stored drag callbacks cannot be disconnected
        without the layer reference — those are handled by the per-layer
        ``disconnect_main_label_sync`` calls that should precede this.
        """
        # --- Label sync handlers ---
        for layer_id, handlers in list(self._sync_handlers.items()):
            for event_emitter, handler in handlers:
                try:
                    event_emitter.disconnect(handler)
                except (ValueError, AttributeError):
                    pass
        self._sync_handlers.clear()

        # --- Main drag handlers ---
        # The per-layer disconnect loop already removed drag callbacks
        # from layer.mouse_drag_callbacks.  Clear any remaining entries.
        self._main_drag_handlers.clear()

    def _create_main_drag_callback(self, label_layer: Labels):
        """Return a mouse_drag generator that syncs main viewer edits to subviewers."""
        def callback(layer, event):
            if layer.mode not in ("paint", "erase", "fill"):
                return
            while event.type != "mouse_release":
                yield
            name = self._base_label_name(layer.name)
            # If normalization created a copy, push data into the shared volume.
            if name in self.shared_labels:
                shared = self.shared_labels[name]
                if shared is not layer.data:
                    shared[:] = layer.data
            self._refresh_subviewer_labels(name)
        return callback

    def _refresh_subviewer_labels(self, layer_name: str) -> None:
        """Refresh all subviewer label layers matching *layer_name*."""
        for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
            for layer in vm.layers:
                if isinstance(layer, Labels) and self._base_label_name(layer.name) == layer_name:
                    layer.refresh()

    def sync_other_views(self, source_name: str, orientation: str) -> None:
        orient = str(orientation).upper()
        base = source_name.replace(f"_{orient}", "")
        if base not in self.shared_labels:
            return
        for vm, o in zip(
            [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz],
            ["XY", "XZ", "YZ"],
        ):
            if o == orient:
                continue
            layer = self._layer_by_name(vm.layers, f"{base}_{o}")
            if layer is not None:
                layer.refresh()
