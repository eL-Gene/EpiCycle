"""Mouse drawing and close-loop fill handler for label subviewers."""

from __future__ import annotations

import time

import numpy as np
from skimage.draw import polygon


class DrawingHandler:
    """Handle paint/erase/fill interactions with optional enclosed fill.

    Uses adaptive throttling based on volume size:
      - Small  (<5M voxels):  50ms
      - Medium (<20M voxels): 100ms
      - Large  (<100M voxels): 200ms
      - Very large (>=100M):  sync only on mouse release
    """

    # Adaptive throttle tiers: (max_voxels, throttle_seconds)
    _THROTTLE_TIERS = [
        (5_000_000, 0.05),
        (20_000_000, 0.10),
        (100_000_000, 0.20),
    ]
    _THROTTLE_DISABLED = 999.0  # effectively sync only on release

    def __init__(self, sync_manager, throttle_seconds: float = 0.08) -> None:
        self.sync_manager = sync_manager
        self.throttle_seconds = max(0.01, float(throttle_seconds))
        self._active_edit_view = None
        self._is_editing = False
        self._polygon_points = []
        # Optional callback fired ONCE on mouse-release BEFORE final sync.
        # Used by 3D curation to snapshot shared_data before the stroke
        # is synced, enabling undo of raw draw corrections.
        self.pre_sync_callback = None
        # Optional callback fired ONCE on mouse-release AFTER final sync.
        # Used by 3D curation to rebuild the isolation proxy so the user
        # sees drawn corrections in the main 3D viewer.
        self.post_sync_callback = None
        # Optional callback fired ONCE after an enclosed-fill polygon
        # is painted AND the auto-increment bump fires.  Argument is
        # the PRE-BUMP ``selected_label`` (the label that was just
        # painted — the one to register in the navigator).  Used by
        # the Curation3DAnnotator (via the controller) to UPSERT a
        # new row in the QC features DataFrame for the freshly-
        # painted cell.  Stays None for de-novo annotation flows
        # that don't drive the navigator.
        self.new_label_callback = None

    def _adaptive_throttle(self, layer) -> float:
        """Pick throttle interval based on volume voxel count."""
        try:
            voxel_count = layer.data.size
        except Exception:
            return self.throttle_seconds
        for max_voxels, interval in self._THROTTLE_TIERS:
            if voxel_count < max_voxels:
                return interval
        return self._THROTTLE_DISABLED

    def create_mouse_drag_callback(self, orientation: str, viewer_model):
        def callback(layer, event):
            return self._on_mouse_drag(layer, event, str(orientation).upper(), viewer_model)

        return callback

    def _on_mouse_drag(self, layer, event, orientation: str, viewer_model):
        if layer.mode not in ("paint", "erase", "fill"):
            return
        # Only respond to LEFT-button drags.  Without this gate, right-
        # click + drag would accumulate polygon_points across mouse_move
        # events and _fill_outline on release would paint an enclosed
        # polygon under the right-drag arc — even though napari's own
        # paint cb is now suppressed via install_button_aware_paint.
        # The EpiCycle drawing-handler enclosed-fill is a second paint
        # path and needs its own button filter.  (Companion to the
        # Mechanism B fix in subviewers.py::install_button_aware_paint.)
        if getattr(event, "button", 1) != 1:
            return

        self._active_edit_view = orientation
        self._is_editing = True
        self._polygon_points = []
        last_sync = 0.0
        throttle = self._adaptive_throttle(layer)

        # Pre-sync: fire BEFORE any sync modifies shared_data.
        # Used by 3D curation to snapshot the label's ROI for undo.
        pre_cb = self.pre_sync_callback
        if pre_cb is not None:
            try:
                pre_cb()
            except Exception:
                pass

        while event.type != "mouse_release":
            if event.type == "mouse_move":
                pos = viewer_model.cursor.position
                z_index = int(pos[0])
                i = int(pos[1])
                j = int(pos[2])
                self._polygon_points.append((i, j, z_index))
                now = time.time()
                if now - last_sync > throttle:
                    self.sync_manager.sync_to_main_view(
                        layer, orientation, viewer_model,
                        defer_main_refresh=True,
                    )
                    last_sync = now
            yield

        enclosed_fill = bool(getattr(layer, "_epicycle_enclosed_fill_enabled", True))
        if len(self._polygon_points) > 2 and layer.mode == "paint" and enclosed_fill:
            self._fill_outline(layer)

            # Capture the painted label NOW — before any auto-increment
            # bump.  This is the label the user just drew with.  We fire
            # new_label_callback below regardless of auto-increment
            # state because the contract is "user painted a label
            # via enclosed-fill, register a navigator row" — independent
            # of whether the brush auto-bumps after.  3D curation mode
            # (Curation3DAnnotator at curation_3d_widget.py:288) sets
            # _auto_increment_labels = False by default; keeping the
            # callback inside that gate would break in production
            # (the entire 3D curation workflow).
            painted_label = 0
            try:
                painted_label = int(getattr(layer, "selected_label", 0))
            except (TypeError, ValueError):
                painted_label = 0

            if bool(getattr(layer, "_epicycle_auto_increment_label", True)):
                try:
                    # 2026-05-28 (fix/step2-autoinc-perf): use the sync
                    # manager's CACHED next-label counter instead of an
                    # np.max scan over every work_labels volume on each
                    # fill (~265ms on busy axial patches).  The counter is
                    # seeded once (one scan) then incremented per fill;
                    # invalidated when predictions/restore introduce labels
                    # (the _insert_slice_mask_with_unique_labels chokepoint).
                    next_label = self.sync_manager.next_fill_label(
                        layer, just_painted=painted_label
                    )
                    layer.selected_label = next_label
                    self.sync_manager.sync_selected_label(layer)
                except Exception:
                    pass

            # Fire new_label_callback unconditionally on enclosed-fill
            # with a non-zero label.  Idempotent on the receiving side
            # (controller's _on_annotator_new_label_painted skips when
            # the label_id is already in the DataFrame, so paint-over
            # of existing labels is a no-op).
            cb = self.new_label_callback
            if cb is not None and painted_label > 0:
                try:
                    cb(painted_label)
                except Exception:
                    pass

        self.sync_manager.track_slice_edit(layer, viewer_model, orientation)
        # Final sync on release — full refresh (main viewer + other views).
        # Layers with _epicycle_defer_main_refresh=True skip the expensive
        # main viewer 3D re-render (e.g. Refine layer during split painting).
        defer_main = bool(getattr(layer, "_epicycle_defer_main_refresh", False))
        self.sync_manager.sync_to_main_view(
            layer, orientation, viewer_model, defer_main_refresh=defer_main,
        )
        self.sync_manager.sync_other_views(layer.name, orientation)
        layer.refresh()
        self._is_editing = False
        # NOTE: _active_edit_view is cleared AFTER post_sync_callback
        # so the callback can read which orientation was drawn on.
        cb = self.post_sync_callback
        if cb is not None:
            try:
                cb()
            except Exception:
                pass
        self._active_edit_view = None

    def _fill_outline(self, layer) -> None:
        if len(self._polygon_points) < 3:
            return
        i_coords = [p[0] for p in self._polygon_points]
        j_coords = [p[1] for p in self._polygon_points]
        z_index = self._polygon_points[0][2]

        rr, cc = polygon(i_coords, j_coords, shape=(layer.data.shape[1], layer.data.shape[2]))
        if bool(getattr(layer, "_epicycle_clip_to_existing", True)):
            # Clip to existing labels (the clip-to-existing rule): the enclosed-fill
            # is a raw write that bypasses napari's preserve_labels, so apply the
            # same rule here — fill ONLY background voxels inside the polygon,
            # never overwrite a neighbouring label.  (Own-label voxels are already
            # the selected label, so leaving them untouched is a no-op.)
            region = layer.data[z_index, rr, cc]
            keep_bg = region == 0
            layer.data[z_index, rr, cc] = np.where(
                keep_bg, layer.selected_label, region
            )
        else:
            # Exempt layer (e.g. the split Refine layer): raw fill so a seed loop
            # can paint OVER the target label being split.
            layer.data[z_index, rr, cc] = layer.selected_label
        self._polygon_points.clear()

    def is_editing(self) -> bool:
        return self._is_editing

    def get_active_edit_view(self):
        return self._active_edit_view
