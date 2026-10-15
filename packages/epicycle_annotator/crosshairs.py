"""Crosshair overlays for the native EpiCycle orthoview annotator."""

from __future__ import annotations

from vispy.scene.visuals import InfiniteLine

COLORS = {
    "cyan": (0.0, 1.0, 1.0, 0.6),
    "magenta": (1.0, 0.0, 1.0, 0.6),
    "yellow": (1.0, 1.0, 0.0, 0.6),
}


class CrosshairOverlay:
    """Two vispy InfiniteLine visuals on one subviewer."""

    def __init__(self, qt_viewer, h_color: str, v_color: str):
        scene = qt_viewer.canvas.view.scene
        self._native = qt_viewer.canvas.native
        self.h_line = InfiniteLine(pos=0, color=COLORS[h_color], vertical=False, parent=scene)
        self.v_line = InfiniteLine(pos=0, color=COLORS[v_color], vertical=True, parent=scene)
        self.h_line.order = 1e6
        self.v_line.order = 1e6

    def set_position(self, h_pos: float, v_pos: float) -> None:
        self.h_line.set_data(pos=h_pos)
        self.v_line.set_data(pos=v_pos)

    def repaint(self) -> None:
        self._native.update()

    def destroy(self) -> None:
        self.h_line.parent = None
        self.v_line.parent = None


class CrosshairManager:
    """Manage synchronized crosshair overlays across XY/XZ/YZ subviewers."""

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
        self._overlays = {}
        self._callbacks = []
        self._updating = False
        self._last_xyz = None

    def toggle_crosshairs(self, enabled: bool) -> None:
        if enabled:
            if "XY" not in self._overlays:
                self._overlays["XY"] = CrosshairOverlay(self.qt_viewer_xy, "cyan", "magenta")
                self._overlays["XZ"] = CrosshairOverlay(self.qt_viewer_xz, "yellow", "magenta")
                self._overlays["YZ"] = CrosshairOverlay(self.qt_viewer_yz, "cyan", "yellow")
            self._attach_listeners()
            self.update_crosshairs()
        else:
            self._detach_listeners()
            for overlay in self._overlays.values():
                overlay.destroy()
            self._overlays.clear()
            self._last_xyz = None

    def _attach_listeners(self) -> None:
        self._detach_listeners()

        def on_step_change(_event=None):
            self.update_crosshairs()

        for qt in [self.qt_viewer_xy, self.qt_viewer_xz, self.qt_viewer_yz]:
            emitter = qt.viewer.dims.events.current_step
            emitter.connect(on_step_change)
            self._callbacks.append((emitter, on_step_change))

    def _detach_listeners(self) -> None:
        for emitter, callback in self._callbacks:
            try:
                emitter.disconnect(callback)
            except Exception:
                continue
        self._callbacks.clear()

    def _safe_current_step0(self, viewer_model) -> int:
        try:
            step = tuple(getattr(viewer_model.dims, "current_step", ()))
            if len(step) > 0:
                return int(step[0])
        except Exception:
            pass
        return 0

    def update_crosshairs(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            z = self._safe_current_step0(self.viewer_model_xy)
            y = self._safe_current_step0(self.viewer_model_xz)
            x = self._safe_current_step0(self.viewer_model_yz)
            xyz = (x, y, z)
            if xyz == self._last_xyz:
                return
            self._last_xyz = xyz

            if "XY" in self._overlays:
                self._overlays["XY"].set_position(y, x)
            if "XZ" in self._overlays:
                self._overlays["XZ"].set_position(z, x)
            if "YZ" in self._overlays:
                self._overlays["YZ"].set_position(y, z)
            for overlay in self._overlays.values():
                overlay.repaint()
        finally:
            self._updating = False

