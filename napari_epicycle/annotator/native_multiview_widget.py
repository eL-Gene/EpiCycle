"""Native EpiCycle multiview annotator widget (Phase J MVP)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
from napari._qt.qt_viewer import QtViewer
from napari.components import ViewerModel
from napari.layers import Image, Labels
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from packages.epicycle_annotator import (
    CrosshairManager,
    DrawingHandler,
    SubviewerMirrorManager,
    LabelSyncManager,
    SubviewerManager,
)


def _safe_deferred_refresh(layer):
    """Sip-deletion-safe wrapper for QTimer-deferred subviewer refresh.

    Phase A #2 (perf push 2026-05-09): subviewer refresh in 3D mode is
    deferred via ``QTimer.singleShot(50, lambda: _safe_deferred_refresh(layer))``.
    If ``layer`` is destroyed within the 50 ms window (annotator close,
    patch switch, layer removal), calling ``layer.refresh()`` would
    raise ``RuntimeError: wrapped C++ object has been deleted`` — a
    process-killing access violation in the QTimer callback (which is
    NOT covered by the surrounding try/except in
    ``_refresh_label_data_in_subviewers`` because it fires on the next
    event-loop tick, outside the original try block).

    Wrapper catches RuntimeError + AttributeError + Exception so the
    deferred callback no-ops cleanly on a deleted/disposed layer.
    """
    try:
        layer.refresh()
    except (RuntimeError, AttributeError, Exception):
        # RuntimeError: sip-deleted Qt shell ("wrapped C++ object ...
        #   has been deleted")
        # AttributeError: defensive — partially-disposed layer
        # Exception: belt-and-suspenders for any other internal failure
        #   (vispy state, etc.) — deferred refresh is best-effort
        pass


class NativeMultiviewAnnotator(QWidget):
    """Three-view annotator focused on draw/close/fill workflow."""

    # ------------------------------------------------------------------
    # setParent override — intercept napari's orphaning
    # ------------------------------------------------------------------
    def setParent(self, parent):
        """Intercept napari's ``setParent(None)`` to hide all child widgets.

        napari's ``remove_dock_widget`` calls ``widget.setParent(None)`` which
        makes the annotator a top-level window.  Its child QWidgets (buttons,
        checkboxes, groups) also become top-level orphans visible on the
        Windows taskbar as "magicgui" (the QApplication name fallback for
        widgets with empty windowTitle).

        The key insight: ``deleteLater()`` + ``sendPostedEvents`` cannot
        destroy children while Python instance attributes (``self._xxx_btn``)
        still reference them.  We must **null out** all Python refs to child
        QWidgets first, then flush deferred deletes.
        """
        if parent is None:
            try:
                self.hide()
            except Exception:
                pass
            # ---- Phase 1: collect all child QWidgets, hide + schedule delete.
            try:
                from qtpy.QtWidgets import QWidget as _QW

                children = self.findChildren(_QW)
                for child in children:
                    try:
                        child.hide()
                        child.deleteLater()
                    except Exception:
                        pass
            except Exception:
                children = []

            # ---- Phase 2: null out ALL instance attributes that reference
            # QWidgets.  Without this, Python ref-counting keeps the C++ objects
            # alive and sendPostedEvents(DeferredDelete) cannot destroy them.
            try:
                from qtpy.QtWidgets import QWidget as _QW2

                for attr_name in list(vars(self)):
                    try:
                        val = getattr(self, attr_name, None)
                        if isinstance(val, _QW2):
                            setattr(self, attr_name, None)
                    except Exception:
                        pass
            except Exception:
                pass

            # ---- Phase 3: flush deferred deletes so children are destroyed
            # NOW, before super().setParent(None) makes us top-level.
            try:
                from qtpy.QtCore import QCoreApplication, QEvent

                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            except Exception:
                pass
        super().setParent(parent)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._active_image_name = ""
        self._active_image_layer_names = []
        self._active_labels_name = ""
        self._active_labels_names = []
        self._managed_layer_names = set()
        self._patch_ids = []
        self._active_patch_id = ""
        self._patch_switch_callback = None
        self._segment_slice_callback = None
        self._overlay_slice_callback = None
        self._accept_slice_callback = None
        self._active_target_mode = "lateral"
        self._3d_curation_mode = False
        self._active_selected_view = "XY"
        self._syncing_slice_slider = {"XY": False, "XZ": False, "YZ": False}
        self._default_draw_mode = "paint"
        self._default_brush_size = 1
        self._default_selected_label = 1
        self._enclosed_fill_enabled = True
        self._auto_increment_labels = True
        # Focused refinement state (K4.4)
        self._focused_mode = False
        self._flagged_label_ids: list = []
        self._flagged_centroids: dict = {}
        self._flagged_nav_index: int = -1
        self._channel_keybindings: list = []
        self.status_label = QLabel("", parent=self)
        self.status_label.setVisible(False)

        # Lifecycle tracker: snapshot all top-level widget IDs BEFORE creating
        # QtViewers.  QtViewer construction can spawn internal top-level
        # windows (QtPopup, magicgui).  Taking the snapshot first ensures those
        # are "new" and eligible for cleanup by _destroy_orphan_toplevel_widgets.
        self._creation_toplevel_ids: set = set()
        try:
            from qtpy.QtWidgets import QApplication as _LifecycleApp
            _lc_app = _LifecycleApp.instance()
            if _lc_app is not None:
                self._creation_toplevel_ids = {
                    id(w) for w in _lc_app.topLevelWidgets() if w is not None
                }
        except Exception:
            pass

        self.viewer_model_xy = ViewerModel()
        self.viewer_model_xz = ViewerModel()
        self.viewer_model_yz = ViewerModel()
        self.qt_viewer_xy = QtViewer(self.viewer_model_xy)
        self.qt_viewer_xz = QtViewer(self.viewer_model_xz)
        self.qt_viewer_yz = QtViewer(self.viewer_model_yz)

        self.subviewer_manager = SubviewerManager(
            self.viewer_model_xy,
            self.viewer_model_xz,
            self.viewer_model_yz,
            self.qt_viewer_xy,
            self.qt_viewer_xz,
            self.qt_viewer_yz,
        )
        self.crosshair_manager = CrosshairManager(
            self.viewer_model_xy,
            self.viewer_model_xz,
            self.viewer_model_yz,
            self.qt_viewer_xy,
            self.qt_viewer_xz,
            self.qt_viewer_yz,
        )
        self.sync_manager = LabelSyncManager(
            self.viewer,
            self.viewer_model_xy,
            self.viewer_model_xz,
            self.viewer_model_yz,
            self.qt_viewer_xy,
            self.qt_viewer_xz,
            self.qt_viewer_yz,
        )
        self.drawing_handler = DrawingHandler(self.sync_manager)
        # Right-click → navigator-focus factory: subclasses (notably
        # Curation3DAnnotator) override _make_subviewer_right_click_handler
        # to return a closure that fires nav.select_label(label_id).
        # Base class returns None so plain native_multiview annotators
        # have no extra subviewer behavior — strictly opt-in for the
        # 3D-curation specialization.
        rc_factory = getattr(self, "_make_subviewer_right_click_handler", None)
        self.layer_manager = SubviewerMirrorManager(
            self.viewer,
            self.subviewer_manager,
            self.sync_manager,
            self.drawing_handler,
            is_managed_layer=self._is_managed_layer,
            right_click_callback_factory=rc_factory,
        )
        self.layer_manager.setup_layer_listeners()

        self._build_ui()
        self._wire_signals()
        self._refresh_layer_options()

    def _layer_by_name(self, layer_list, name: str):
        if not name:
            return None
        try:
            if name in layer_list:
                return layer_list[name]
        except Exception:
            pass
        for layer in layer_list:
            if getattr(layer, "name", "") == name:
                return layer
        return None

    def _is_managed_layer(self, layer) -> bool:
        name = str(getattr(layer, "name", "")).strip()
        return bool(name) and name in self._managed_layer_names

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        yz_box = self._build_view_panel(self.qt_viewer_yz, orientation="YZ")
        xy_box = self._build_view_panel(self.qt_viewer_xy, orientation="XY")
        xz_box = self._build_view_panel(self.qt_viewer_xz, orientation="XZ")
        utility_box = self._build_utility_panel()

        # Minimum widths for the left column; no max so the user can drag wider.
        yz_box.setMinimumWidth(120)
        utility_box.setMinimumWidth(170)

        # Top row: YZ (narrow) + XY (wide)
        top_row = QSplitter(Qt.Horizontal)
        top_row.addWidget(yz_box)
        top_row.addWidget(xy_box)
        top_row.setSizes([180, 500])
        top_row.setStretchFactor(0, 1)
        top_row.setStretchFactor(1, 5)
        top_row.setChildrenCollapsible(False)

        # Bottom row: utility panel (narrow) + XZ (wide)
        bottom_row = QSplitter(Qt.Horizontal)
        bottom_row.addWidget(utility_box)
        bottom_row.addWidget(xz_box)
        bottom_row.setSizes([180, 500])
        bottom_row.setStretchFactor(0, 1)
        bottom_row.setStretchFactor(1, 5)
        bottom_row.setChildrenCollapsible(False)

        # Vertical splitter: top row dominates (5:2 ratio)
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_row)
        main_splitter.addWidget(bottom_row)
        main_splitter.setSizes([500, 200])
        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setChildrenCollapsible(False)

        layout.addWidget(main_splitter, 1)
        self._hide_builtin_dim_controls()

    def _build_view_panel(self, widget: QWidget, *, orientation: str) -> QWidget:
        panel = QFrame()
        slider_orientation = Qt.Vertical if orientation == "YZ" else Qt.Horizontal
        slider = QSlider(slider_orientation)
        handle_color = {
            "YZ": "#ff4fd8",  # magenta
            "XY": "#f3e55b",  # yellow
            "XZ": "#52e8ff",  # cyan
        }.get(str(orientation).upper(), "#6ea7ff")
        slider.setRange(0, 0)
        slider.setValue(0)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        slider.setVisible(False)
        slider.setToolTip(f"{orientation} slice")
        slider_style = """
            QSlider::groove:horizontal {
                height: 4px;
                margin: 0px;
                background: #5b6572;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 10px;
                margin: -4px 0px;
                background: __HANDLE_COLOR__;
                border-radius: 2px;
            }
            QSlider::groove:vertical {
                width: 4px;
                margin: 0px;
                background: #5b6572;
                border-radius: 2px;
            }
            QSlider::handle:vertical {
                height: 10px;
                margin: 0px -4px;
                background: __HANDLE_COLOR__;
                border-radius: 2px;
            }
            """
        slider.setStyleSheet(slider_style.replace("__HANDLE_COLOR__", handle_color))
        if slider_orientation == Qt.Vertical:
            outer = QHBoxLayout(panel)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            slider.setMinimumHeight(120)
            slider.setFixedWidth(10)
            # Requested: keep YZ slider on the left side.
            outer.addWidget(slider, 0)
            outer.addWidget(widget, 1)
        else:
            outer = QVBoxLayout(panel)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            outer.addWidget(widget, 1)
            slider.setFixedHeight(10)
            outer.addWidget(slider, 0)
        if orientation == "XY":
            self.xy_slice_slider = slider
        elif orientation == "XZ":
            self.xz_slice_slider = slider
        else:
            self.yz_slice_slider = slider
        return panel

    def _build_data_group(self) -> QWidget:
        group = QGroupBox("Data Source")
        form = QFormLayout(group)

        self.image_combo = QComboBox()
        self.labels_combo = QComboBox()
        self.refresh_layers_btn = QPushButton("Refresh Layers")
        self.create_labels_btn = QPushButton("Create Labels From Image")
        self.sync_to_subviewers_btn = QPushButton("Load Into Subviewers")

        row_buttons = QHBoxLayout()
        row_buttons.addWidget(self.refresh_layers_btn)
        row_buttons.addWidget(self.create_labels_btn)

        form.addRow("Image layer:", self.image_combo)
        form.addRow("Labels layer:", self.labels_combo)
        form.addRow(row_buttons)
        form.addRow(self.sync_to_subviewers_btn)
        return group

    def _build_utility_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        draw_group = self._build_draw_group()
        self.data_group = self._build_data_group()
        self.data_group.setVisible(False)
        layout.addWidget(draw_group, 1)
        # Keep data-group instantiated for compatibility/debug plumbing, but
        # hide it from quick annotation UX by default.
        return panel

    def _build_draw_group(self) -> QWidget:
        group = QGroupBox("Quick Annotate")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(6)

        self.crosshairs_check = QCheckBox("Crosshair")
        self.crosshairs_check.setToolTip("Show crosshairs")
        self.crosshairs_check.setChecked(True)
        self.patch_context_combo = QComboBox()
        self.patch_context_combo.setMinimumWidth(100)
        self.switch_patch_btn = QPushButton("Switch")
        self.switch_patch_btn.setToolTip("Switch to selected patch context")
        self.segment_mode_btn = QPushButton("Segment Lateral (XY)")
        self.accept_slice_btn = QPushButton("Accept Current Slice")
        self.accept_slice_btn.setToolTip("Register the current slice annotation as accepted for training.")

        patch_row = QHBoxLayout()
        patch_row.setSpacing(4)
        patch_row.addWidget(QLabel("Patch"))
        patch_row.addWidget(self.patch_context_combo, 1)
        patch_row.addWidget(self.switch_patch_btn)

        self.overlay_mode_combo = QComboBox()
        self.overlay_mode_combo.addItems(["Off", "Probability", "Flow"])
        self.overlay_mode_combo.setCurrentText("Off")
        self.overlay_mode_combo.setToolTip("Subviewer-only overlay from cached segmentation outputs.")
        self.overlay_opacity_sb = QSpinBox()
        self.overlay_opacity_sb.setRange(0, 100)
        self.overlay_opacity_sb.setValue(55)
        self.overlay_opacity_sb.setSuffix("%")
        self.overlay_opacity_sb.setToolTip("Overlay opacity")

        # 2026-06-01: crosshair sits in its own QHBoxLayout so subclasses
        # can append section-level controls inline (e.g. Curation2DAnnotator
        # appends a help-button via help_button() from ui_shell.theme).
        # Layout exposed as `self._crosshair_row_layout`.
        self._crosshair_row_layout = QHBoxLayout()
        self._crosshair_row_layout.setContentsMargins(0, 0, 0, 0)
        self._crosshair_row_layout.setSpacing(4)
        self._crosshair_row_layout.addWidget(self.crosshairs_check)
        self._crosshair_row_layout.addStretch(1)
        layout.addLayout(self._crosshair_row_layout)
        layout.addLayout(patch_row)
        layout.addWidget(self.segment_mode_btn)
        layout.addWidget(self.accept_slice_btn)
        overlay_form = QFormLayout()
        overlay_form.setContentsMargins(0, 0, 0, 0)
        overlay_form.setHorizontalSpacing(6)
        overlay_form.setVerticalSpacing(4)
        overlay_form.addRow("Overlay:", self.overlay_mode_combo)
        overlay_form.addRow("Opacity:", self.overlay_opacity_sb)
        layout.addLayout(overlay_form)

        self.open_advanced_features_btn = QPushButton("Advanced Features...")
        self.open_advanced_features_btn.setToolTip(
            "Open SAM, classic 2D methods, and multiviewer options"
        )
        layout.addWidget(self.open_advanced_features_btn)

        self._open_multiviewer_settings_btn = QPushButton("Multiviewer Settings...")
        self._open_multiviewer_settings_btn.setToolTip(
            "Open multiviewer display settings (crosshairs, sync)"
        )
        layout.addWidget(self._open_multiviewer_settings_btn)
        # release hide list 2026-10-13 (dock controls are unwired)
        self._open_multiviewer_settings_btn.setVisible(False)

        # Focused refinement controls (K4.4)
        self._focused_mode_group = QGroupBox("Refinement Navigation")
        fm_lay = QVBoxLayout(self._focused_mode_group)
        fm_lay.setContentsMargins(4, 4, 4, 4)
        fm_lay.setSpacing(4)
        self._focused_toggle_btn = QPushButton("Focused Mode: OFF")
        self._focused_toggle_btn.setCheckable(True)
        self._focused_toggle_btn.setToolTip(
            "Toggle focused mode: only flagged labels visible, others dimmed"
        )
        # 2026-06-01: toggle row wrapped in QHBoxLayout so subclasses can
        # append a help-button inline (mirrors the crosshair_row pattern).
        self._focused_toggle_row_layout = QHBoxLayout()
        self._focused_toggle_row_layout.setContentsMargins(0, 0, 0, 0)
        self._focused_toggle_row_layout.setSpacing(4)
        self._focused_toggle_row_layout.addWidget(self._focused_toggle_btn, 1)
        fm_lay.addLayout(self._focused_toggle_row_layout)
        nav_row = QHBoxLayout()
        self._prev_flagged_btn = QPushButton("\u25c0 Prev")
        self._next_flagged_btn = QPushButton("Next \u25b6")
        nav_row.addWidget(self._prev_flagged_btn)
        nav_row.addWidget(self._next_flagged_btn)
        fm_lay.addLayout(nav_row)
        self._focused_nav_label = QLabel("")
        self._focused_nav_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._focused_nav_label.setWordWrap(True)
        fm_lay.addWidget(self._focused_nav_label)
        self._focused_mode_group.setVisible(False)
        layout.addWidget(self._focused_mode_group)

        layout.addStretch(1)
        return group

    def _build_info_group(self) -> QWidget:
        group = QGroupBox("Logs")
        lay = QVBoxLayout(group)
        visual_status = QLabel("Logs hidden in compact mode.")
        visual_status.setWordWrap(True)
        lay.addWidget(visual_status)
        group.setMaximumHeight(120)
        return group

    def _hide_builtin_dim_controls(self) -> None:
        """Hide napari's default UI chrome (dims, buttons, controls) on subviewers.

        Subviewers are display-only — they don't need napari's layer list,
        layer buttons, viewer buttons, layer controls, or dims sliders.
        Leaving them visible causes UI occlusion in the subviewer panels.
        """
        for qt_viewer in [self.qt_viewer_xy, self.qt_viewer_xz, self.qt_viewer_yz]:
            # Hide napari's built-in UI panels that occlude the subviewers.
            for attr in (
                "dims",              # Dims slider row
                "layerButtons",      # Add/remove/up/down layer buttons
                "viewerButtons",     # ndisplay, grid, etc. viewer buttons
                "dockLayerControls", # Layer property controls dock
                "dockLayerList",     # Layer list dock
                "controls",          # Layer controls panel
            ):
                try:
                    widget = getattr(qt_viewer, attr, None)
                    if widget is not None:
                        widget.setVisible(False)
                        widget.setMaximumHeight(0)
                except Exception:
                    pass

            # Fallback: hide individual dim slider widgets by class name.
            try:
                qt_dims = getattr(qt_viewer, "dims", None)
                displayed = list(getattr(qt_dims, "_displayed_sliders", []) or [])
                for slider_widget in displayed:
                    try:
                        slider_widget.setVisible(False)
                    except Exception:
                        continue
            except Exception:
                pass

            for child in qt_viewer.findChildren(QWidget):
                cls = str(child.__class__.__name__)
                if "DimSliderWidget" in cls:
                    try:
                        child.setVisible(False)
                    except Exception:
                        continue

    def _infer_xy_slice_index_from_name(self, label_name: str, max_z: int, fallback: int) -> int:
        text = str(label_name or "")
        match = re.search(r"_XY_(\d+)", text)
        if match:
            try:
                idx = int(match.group(1))
                return max(0, min(max_z - 1, idx))
            except Exception:
                pass
        return max(0, min(max_z - 1, int(fallback)))

    def _normalize_image_volume(self, image_data):
        arr = np.asarray(image_data)
        if arr.ndim == 2:
            return arr[np.newaxis, ...], "Promoted 2D image to single-slice 3D volume."
        if arr.ndim == 3:
            return arr, ""
        if arr.ndim == 4 and arr.shape[-1] in {3, 4}:
            return arr, ""
        raise ValueError(f"Image must be 2D or 3D, got ndim={arr.ndim}.")

    def _normalize_labels_volume(self, labels_data, image_shape, label_name: str):
        arr = np.asarray(labels_data)
        spatial_shape = tuple(image_shape[:3]) if len(tuple(image_shape)) == 4 else tuple(image_shape)
        if arr.ndim == 3:
            if tuple(arr.shape) != spatial_shape:
                raise ValueError(
                    f"3D labels shape {tuple(arr.shape)} does not match image spatial shape {spatial_shape}."
                )
            return arr.astype(np.int32, copy=False), ""

        if arr.ndim == 2:
            expected_xy = tuple(spatial_shape[-2:])
            if tuple(arr.shape) != expected_xy:
                raise ValueError(
                    f"2D labels shape {tuple(arr.shape)} does not match image XY shape {expected_xy}."
                )
            fallback = 0
            try:
                dims = getattr(self.viewer, "dims", None)
                if dims is not None and len(getattr(dims, "current_step", ())) > 0:
                    fallback = int(dims.current_step[0])
            except Exception:
                fallback = 0
            z = self._infer_xy_slice_index_from_name(label_name, int(spatial_shape[0]), fallback)
            out = np.zeros(spatial_shape, dtype=np.int32)
            out[z, :, :] = arr.astype(np.int32, copy=False)
            return out, f"Promoted 2D labels to 3D volume at Z={z}."

        raise ValueError(f"Labels must be 2D or 3D, got ndim={arr.ndim}.")

    def _wire_signals(self) -> None:
        self.refresh_layers_btn.clicked.connect(self._refresh_layer_options)
        self.create_labels_btn.clicked.connect(self._create_labels_from_image)
        self.sync_to_subviewers_btn.clicked.connect(self._load_selected_layers)
        self.switch_patch_btn.clicked.connect(self._switch_patch_context_from_combo)
        self.segment_mode_btn.clicked.connect(self._request_segment_active_mode)
        self.accept_slice_btn.clicked.connect(self._request_accept_active_slice)
        self.crosshairs_check.toggled.connect(self._toggle_crosshairs)
        self.overlay_mode_combo.currentTextChanged.connect(lambda *_: self._refresh_overlays_all())
        self.overlay_opacity_sb.valueChanged.connect(lambda *_: self._refresh_overlays_all())
        self.open_advanced_features_btn.clicked.connect(self._open_advanced_features)
        self._open_multiviewer_settings_btn.clicked.connect(
            self._open_multiviewer_settings
        )
        # Focused refinement (K4.4)
        self._focused_toggle_btn.toggled.connect(self._toggle_focused_mode)
        self._prev_flagged_btn.clicked.connect(self._navigate_prev_flagged)
        self._next_flagged_btn.clicked.connect(self._navigate_next_flagged)

        # --- Viewer-level event handlers (napari EventEmitter) ---
        # Store references for clean disconnection in _disconnect_viewer_events.
        self._viewer_inserted_cb = lambda _e: self._refresh_layer_options()
        self._viewer_removed_cb = lambda _e: self._refresh_layer_options()
        self._dims_xy_cb = lambda _event=None: self._sync_slider_from_dims("XY")
        self._dims_xz_cb = lambda _event=None: self._sync_slider_from_dims("XZ")
        self._dims_yz_cb = lambda _event=None: self._sync_slider_from_dims("YZ")

        self.viewer.layers.events.inserted.connect(self._viewer_inserted_cb)
        self.viewer.layers.events.removed.connect(self._viewer_removed_cb)
        self.viewer.layers.events.removed.connect(self._on_main_layer_removed)
        self.viewer.layers.selection.events.active.connect(self.sync_manager.sync_active_layer)
        for orient, slider in [
            ("XY", getattr(self, "xy_slice_slider", None)),
            ("XZ", getattr(self, "xz_slice_slider", None)),
            ("YZ", getattr(self, "yz_slice_slider", None)),
        ]:
            if slider is not None:
                slider.valueChanged.connect(
                    lambda value, o=orient: self._on_slice_slider_changed(o, value)
                )
        self.viewer_model_xy.dims.events.current_step.connect(self._dims_xy_cb)
        self.viewer_model_xz.dims.events.current_step.connect(self._dims_xz_cb)
        self.viewer_model_yz.dims.events.current_step.connect(self._dims_yz_cb)

    def _disconnect_viewer_events(self) -> None:
        """Disconnect napari viewer-level event handlers to prevent
        stale callbacks after the widget is destroyed.

        Also resets per-layer guard flags so a new widget instance can
        reconnect to the same layers.
        """
        # LayerLifecycleManager installs global listeners on the main viewer.
        # These must be disconnected on teardown, or callbacks accumulate
        # across close/relaunch cycles and progressively degrade performance.
        try:
            self.layer_manager.disconnect_layer_listeners()
        except Exception:
            pass

        # --- Rename handlers on managed layers ---
        # Connected in _load_selected_layers(); must be disconnected here
        # to prevent stale callbacks accumulating across close/relaunch.
        self._disconnect_rename_handlers()

        # --- Viewer-level events ---
        try:
            self.viewer.layers.events.inserted.disconnect(self._viewer_inserted_cb)
        except (ValueError, AttributeError):
            pass
        try:
            self.viewer.layers.events.removed.disconnect(self._viewer_removed_cb)
        except (ValueError, AttributeError):
            pass
        try:
            self.viewer.layers.events.removed.disconnect(self._on_main_layer_removed)
        except (ValueError, AttributeError):
            pass
        try:
            self.viewer.layers.selection.events.active.disconnect(
                self.sync_manager.sync_active_layer
            )
        except (ValueError, AttributeError):
            pass
        # Standalone mode auto-load listener (set by launcher._setup_standalone_auto_load)
        standalone_cb = getattr(self, "_standalone_layer_inserted_cb", None)
        if standalone_cb is not None:
            try:
                self.viewer.layers.events.inserted.disconnect(standalone_cb)
            except (ValueError, AttributeError):
                pass
            self._standalone_layer_inserted_cb = None
        # Subviewer dims events
        for cb, vm in [
            (getattr(self, "_dims_xy_cb", None), self.viewer_model_xy),
            (getattr(self, "_dims_xz_cb", None), self.viewer_model_xz),
            (getattr(self, "_dims_yz_cb", None), self.viewer_model_yz),
        ]:
            if cb is not None:
                try:
                    vm.dims.events.current_step.disconnect(cb)
                except (ValueError, AttributeError):
                    pass

        # --- Reset per-layer guard flags ---
        # These flags are set on the layer objects themselves (not the widget)
        # to prevent duplicate handler connections.  When the widget is
        # destroyed, the flags must be cleared so a fresh widget can
        # reconnect to the same layer.
        lbl_name = getattr(self, "_active_labels_name", "")
        if lbl_name:
            try:
                layer = self.viewer.layers[lbl_name]
                for flag in (
                    "_epicycle_pin_connected",
                    "_epicycle_labels_update_connected",
                    "_epicycle_show_sel_connected",
                ):
                    if hasattr(layer, flag):
                        setattr(layer, flag, False)
            except (KeyError, Exception):
                pass

        # --- Disconnect sync/subviewer handlers on ALL viewer layers ---
        # Previous approach: name-based loops only covered tracked names,
        # then a "safety net" reset guard flags WITHOUT disconnecting.
        # Result: old handlers survived + new handlers connected = accumulation.
        #
        # Fix: loop ALL viewer layers and call manager disconnect methods on
        # each.  The disconnect methods look up by id(layer) in their handler
        # dicts — if the layer has no entry, the call is a safe no-op.
        # This catches renamed layers, dynamically added layers, and any
        # layers the name lists missed.
        sync = getattr(self, "sync_manager", None)
        submgr = getattr(self, "subviewer_manager", None)
        try:
            for layer in list(self.viewer.layers):
                if sync is not None:
                    try:
                        sync.disconnect_label_sync(layer)
                    except Exception:
                        pass
                    try:
                        sync.disconnect_main_label_sync(layer)
                    except Exception:
                        pass
                if submgr is not None:
                    try:
                        submgr.disconnect_layer_visibility(layer)
                    except Exception:
                        pass
                    try:
                        submgr.disconnect_image_style(layer)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Belt-and-suspenders: flush manager handler dicts ---
        # Catches entries for layers that were garbage collected between
        # connect and disconnect (no longer in self.viewer.layers).
        if sync is not None:
            try:
                sync.disconnect_all()
            except Exception:
                pass
        if submgr is not None:
            try:
                submgr.disconnect_all()
            except Exception:
                pass

        # --- Reset ALL epicycle guard flags on ALL layers ---
        # After the disconnect loops above, flags should already be False.
        # This sweep catches edge cases where disconnect raised but the flag
        # wasn't reset, and also handles curation-specific flags
        # (_epicycle_pin_connected, etc.) that are managed by the subclass
        # rather than by sync/subviewer managers.
        try:
            for layer in self.viewer.layers:
                for flag in (
                    "_epicycle_sync_connected",
                    "_epicycle_main_drag_connected",
                    "_epicycle_vis_synced",
                    "_epicycle_style_synced",
                    "_epicycle_pin_connected",
                    "_epicycle_labels_update_connected",
                    "_epicycle_show_sel_connected",
                ):
                    if hasattr(layer, flag):
                        setattr(layer, flag, False)
        except Exception:
            pass

    def _close_orphan_magicgui_windows(self) -> None:
        """Best-effort cleanup for orphan top-level magicgui popups.

        Some QtViewer close paths leave detached top-level windows titled
        ``magicgui``. When they accumulate across relaunches, the UI becomes
        increasingly sluggish.
        """
        try:
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QCoreApplication, QEvent
        except Exception:
            return
        app = QApplication.instance()
        if app is None:
            return
        try:
            top_levels = list(app.topLevelWidgets())
        except Exception:
            return
        for widget in top_levels:
            if widget is None:
                continue
            try:
                title = str(widget.windowTitle() or "").strip().lower()
            except Exception:
                title = ""
            try:
                cls = str(widget.__class__.__name__ or "").strip().lower()
            except Exception:
                cls = ""
            try:
                obj_name = str(widget.objectName() or "").strip().lower()
            except Exception:
                obj_name = ""
            # Match magicgui AND orphan QtPopup/QtModalPopup windows.
            is_orphan = (
                "magicgui" in title or "magicgui" in cls or "magicgui" in obj_name
                or "qtmodalpopup" in obj_name or "qtpopup" in cls
            )
            if not is_orphan:
                continue
            try:
                widget.hide()
            except Exception:
                pass
            try:
                widget.close()
            except Exception:
                pass
            # Skip unparenting — it can orphan child widgets during teardown.
            try:
                widget.deleteLater()
            except Exception:
                pass
        try:
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        except Exception:
            pass

    # Known orphan widget types that accumulate across close/relaunch cycles.
    # Only these get destroyed — UI controls (QPushButton, QCheckBox, etc.)
    # are our own widgets and get cleaned up naturally by Qt parent hierarchy.
    _ORPHAN_CLASSES = frozenset({
        "QFrame",       # napari QtViewer internal frames
        "QtPopup",      # napari dims play button popups
        "QtModalPopup", # napari dims modal popups
        "QWidget",      # blank/top-level wrappers frequently leaked by QtViewer close
    })

    def _destroy_orphan_toplevel_widgets(self) -> None:
        """Destroy orphan top-level widgets created during this widget's lifetime.

        Uses the snapshot from ``__init__`` (``_creation_toplevel_ids``) to
        identify new widgets, then destroys known orphan classes
        (QFrame, QtPopup, QtModalPopup, QWidget).  Also destroys any
        widget with ``magicgui`` in title/class/objectName regardless of class.
        UI controls like QPushButton, QCheckBox, etc. are our own widgets and
        are left for Qt's natural
        parent-based cleanup.

        Safe to call multiple times (idempotent) and from a deferred
        QTimer callback.
        """
        pre_ids = getattr(self, "_creation_toplevel_ids", None)
        if not pre_ids:
            return
        try:
            from qtpy.QtWidgets import QApplication
        except Exception:
            return
        app = QApplication.instance()
        if app is None:
            return
        _safe_ids = set(pre_ids)
        _safe_ids.add(id(self))
        try:
            qt_main = getattr(getattr(self.viewer, "window", None), "_qt_window", None)
            if qt_main is not None:
                _safe_ids.add(id(qt_main))
        except Exception:
            pass
        destroyed = 0
        for w in list(app.topLevelWidgets()):
            if w is None or id(w) in _safe_ids:
                continue
            cls_name = type(w).__name__
            cls_name_l = str(cls_name or "").lower()
            try:
                _obj = str(w.objectName() or "").lower()
            except Exception:
                _obj = ""
            try:
                _title = str(w.windowTitle() or "").lower()
            except Exception:
                _title = ""
            is_magicgui_name = (
                "magicgui" in _title
                or "magicgui" in cls_name_l
                or "magicgui" in _obj
            )
            if not is_magicgui_name and cls_name not in self._ORPHAN_CLASSES:
                continue
            destroyed += 1
            try:
                w.hide()
                w.close()
                w.deleteLater()
            except Exception:
                pass
        if destroyed > 0:
            try:
                from qtpy.QtCore import QCoreApplication, QEvent
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            except Exception:
                pass
            pass  # orphan widgets destroyed

    def _sweep_magicgui_only(self) -> None:
        """Close orphan magicgui windows WITHOUT touching QtPopup/QtModalPopup.

        QtPopup/QtModalPopup windows belong to the CURRENT subviewers' dims
        play buttons — destroying them forces expensive recreation (3-7s lag).
        This targeted sweep only catches magicgui-framework orphans that
        accumulate across close/relaunch cycles.
        """
        try:
            from qtpy.QtWidgets import QApplication
        except Exception:
            return
        app = QApplication.instance()
        if app is None:
            return
        closed = 0
        try:
            for w in list(app.topLevelWidgets()):
                if w is None:
                    continue
                cls_name = type(w).__name__.lower()
                try:
                    obj_name = str(w.objectName() or "").lower()
                except Exception:
                    obj_name = ""
                try:
                    title = str(w.windowTitle() or "").lower()
                except Exception:
                    title = ""
                # ONLY match magicgui — skip qtpopup/qtmodalpopup (current subviewers).
                is_magicgui = (
                    "magicgui" in title
                    or "magicgui" in cls_name
                    or "magicgui" in obj_name
                )
                if not is_magicgui:
                    continue
                closed += 1
                try:
                    w.hide()
                    w.close()
                    # Skip unparenting — it can orphan child widgets.
                    w.deleteLater()
                except Exception:
                    pass
        except Exception:
            pass
        if closed > 0:
            try:
                from qtpy.QtCore import QCoreApplication, QEvent
                # Only flush deferred deletes — do NOT call processEvents()
                # because it can materialize NEW magicgui windows from pending
                # events, creating the exact problem we're trying to solve.
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            except Exception:
                pass
            pass  # magicgui orphans closed

    def _close_subviewers(self) -> None:
        """Close the 3 QtViewer instances to prevent orphaned child windows.

        napari's QtViewer creates internal top-level ``QtPopup`` windows
        (children of the dims play buttons) that survive ``qtv.close()``
        alone.  Full cleanup requires stopping and closing the dims widget
        first — this destroys the QtPopup windows — then closing the
        QtViewer itself.  Avoid manual unparenting during teardown: reparenting
        to ``None`` can create transient top-level windows ("magicgui") before
        deferred deletion runs.
        """
        try:
            self.crosshair_manager.toggle_crosshairs(False)
        except Exception:
            pass
        try:
            self.subviewer_manager.clear()
        except Exception:
            pass
        for qtv in [self.qt_viewer_xy, self.qt_viewer_xz, self.qt_viewer_yz]:
            # Explicitly destroy orphan-prone children BEFORE closing the
            # QtViewer.  Targets: QtPopup/QtModalPopup (dims play buttons)
            # AND any magicgui-framework widgets that become detached
            # top-level windows when the QtViewer is orphaned.
            try:
                from qtpy.QtWidgets import QWidget
                for child in list(qtv.findChildren(QWidget)):
                    cls_name = type(child).__name__.lower()
                    try:
                        child_title = str(child.windowTitle() or "").lower()
                    except Exception:
                        child_title = ""
                    try:
                        child_obj = str(child.objectName() or "").lower()
                    except Exception:
                        child_obj = ""
                    is_orphan_candidate = (
                        "qtpopup" in cls_name
                        or "qtmodalpopup" in cls_name
                        or "magicgui" in child_title
                        or "magicgui" in cls_name
                        or "magicgui" in child_obj
                    )
                    if is_orphan_candidate:
                        try:
                            child.hide()
                            child.close()
                            child.deleteLater()
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                # Stop dims animation thread (prevents segfault on close)
                qtv.dims.stop()
                # Close dims widget — destroys QtPopup top-level children
                qtv.dims.close()
            except Exception:
                pass
            try:
                qtv.close()
                qtv.deleteLater()
            except Exception:
                pass
        # Flush deferred deletes only — do NOT call processEvents() during
        # teardown as it can materialize new magicgui windows from pending events.
        try:
            from qtpy.QtCore import QCoreApplication, QEvent
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        except Exception:
            pass

        # Lifecycle cleanup: destroy ALL top-level widgets that didn't exist
        # when this widget was created.  This is the definitive catch-all for
        # magicgui and other orphan windows that accumulate across relaunch
        # cycles.  Name-matching sweeps miss windows whose class/title/objectName
        # don't contain "magicgui" — this approach doesn't depend on names.
        self._destroy_orphan_toplevel_widgets()

        # Schedule a deferred sweep for the NEXT event loop tick.  Some orphan
        # windows are created by Qt DURING closeEvent processing (e.g., napari
        # dock widget removal triggers magicgui creation).  The deferred sweep
        # catches those since they don't exist yet at this point.
        try:
            from qtpy.QtCore import QTimer as _CloseTimer
            _CloseTimer.singleShot(0, self._destroy_orphan_toplevel_widgets)
            # Late orphan windows can appear after closeEvent completes.
            # Run name-based magicgui sweeps on subsequent ticks as a safety net.
            for _delay in (0, 75, 200, 500):
                _CloseTimer.singleShot(_delay, self._close_orphan_magicgui_windows)
        except Exception:
            pass
        try:
            from napari_epicycle.annotator.launcher import (
                _schedule_close_orphan_sweep_retries,
            )

            _schedule_close_orphan_sweep_retries()
        except Exception:
            pass

    def set_context(
        self,
        *,
        image_layer_name: str = "",
        labels_layer_name: str = "",
        auto_load: bool = True,
        target_view: str = "",
        selected_view: str = "",
    ) -> None:
        """Bind UI to a specific image/labels pair and optionally auto-load."""
        if target_view:
            self._active_target_mode = "lateral" if str(target_view).lower().startswith("lat") else "axial"
        if selected_view:
            self._active_selected_view = str(selected_view).upper()
        self._refresh_segment_button_text()
        if image_layer_name:
            self._refresh_layer_options()
            if self.image_combo.findText(image_layer_name) >= 0:
                self.image_combo.setCurrentText(image_layer_name)
        if labels_layer_name:
            self._refresh_layer_options()
            if self.labels_combo.findText(labels_layer_name) >= 0:
                self.labels_combo.setCurrentText(labels_layer_name)
            patch = self._labels_patch_prefix(labels_layer_name)
            if patch:
                self._active_patch_id = patch
                self._set_patch_combo_to_active()
        if auto_load and self.image_combo.currentText() and self.labels_combo.currentText():
            self._load_selected_layers()

    def set_patch_options(self, *, patch_ids, active_patch_id="", switch_patch_callback=None, patch_statuses=None):
        vals = []
        for p in (patch_ids or []):
            text = str(p or "").strip()
            if not text:
                continue
            vals.append(text)
        seen = set()
        ordered = []
        for p in vals:
            if p in seen:
                continue
            seen.add(p)
            ordered.append(p)
        self._patch_ids = ordered
        self._patch_statuses = patch_statuses or {}
        if active_patch_id:
            self._active_patch_id = str(active_patch_id).strip()
        self._patch_switch_callback = switch_patch_callback
        self._populate_patch_context_combo()

    def set_runtime_callbacks(self, *, segment_slice_callback=None, overlay_slice_callback=None, accept_slice_callback=None, navigate_to_slice_callback=None):
        self._segment_slice_callback = segment_slice_callback
        self._overlay_slice_callback = overlay_slice_callback
        self._accept_slice_callback = accept_slice_callback
        self._navigate_to_slice_callback = navigate_to_slice_callback

    def set_3d_curation_mode(self, enabled: bool = True) -> None:
        """Enable/disable 3D curation mode.

        When enabled, all three orthogonal views are editable on the same
        labels volume (no lateral/axial constraint). Used by the
        Reconstruct + Curate 3D workflow.
        """
        self._3d_curation_mode = bool(enabled)
        if enabled:
            self._active_target_mode = "3d_curation"
        self._apply_edit_constraints()

    def set_focused_refinement(
        self,
        flagged_ids: list,
        centroids: dict,
    ) -> None:
        """Configure focused refinement with flagged label IDs and centroids.

        Parameters
        ----------
        flagged_ids : list of int
            Label IDs that are QC-flagged.
        centroids : dict
            {label_id: (z, y, x)} centroid coordinates.
        """
        self._flagged_label_ids = list(flagged_ids)
        self._flagged_centroids = dict(centroids)
        self._flagged_nav_index = -1
        has_flags = bool(flagged_ids)
        self._focused_mode_group.setVisible(
            has_flags and self._3d_curation_mode
        )
        if has_flags:
            self._focused_nav_label.setText(
                f"{len(flagged_ids)} flagged labels"
            )

    def _toggle_focused_mode(self, enabled: bool) -> None:
        """Toggle between focused (flagged-only) and free (all-visible) modes."""
        self._focused_mode = bool(enabled)
        if enabled:
            self._focused_toggle_btn.setText("Focused Mode: ON")
            self._apply_focused_visibility()
        else:
            self._focused_toggle_btn.setText("Focused Mode: OFF")
            self._clear_focused_visibility()

    def _apply_focused_visibility(self) -> None:
        """Set non-flagged labels to near-transparent in subviewer labels layers."""
        flagged_set = set(self._flagged_label_ids)
        for layer in self._all_subviewer_label_layers():
            try:
                unique_labels = set(int(v) for v in np.unique(layer.data)) - {0}
                dim_colors = {}
                for lid in unique_labels:
                    if lid not in flagged_set:
                        dim_colors[int(lid)] = (0.5, 0.5, 0.5, 0.05)
                if dim_colors:
                    layer.color = dim_colors
            except Exception:
                pass

    def _clear_focused_visibility(self) -> None:
        """Restore all labels to default visibility."""
        for layer in self._all_subviewer_label_layers():
            try:
                layer.color = {}
            except Exception:
                pass

    def _navigate_next_flagged(self) -> None:
        """Navigate to the next flagged label centroid."""
        if not self._flagged_label_ids:
            return
        self._flagged_nav_index = (
            (self._flagged_nav_index + 1) % len(self._flagged_label_ids)
        )
        self._jump_to_flagged(self._flagged_nav_index)

    def _navigate_prev_flagged(self) -> None:
        """Navigate to the previous flagged label centroid."""
        if not self._flagged_label_ids:
            return
        self._flagged_nav_index = (
            (self._flagged_nav_index - 1) % len(self._flagged_label_ids)
        )
        self._jump_to_flagged(self._flagged_nav_index)

    def _jump_to_flagged(self, index: int) -> None:
        """Jump all three subviewer sliders to the centroid of the flagged label."""
        if index < 0 or index >= len(self._flagged_label_ids):
            return
        label_id = self._flagged_label_ids[index]
        centroid = self._flagged_centroids.get(label_id)
        if centroid is None:
            return
        z, y, x = centroid
        self.navigate_to_slice("XY", z)
        self.navigate_to_slice("XZ", y)
        self.navigate_to_slice("YZ", x)
        self._focused_nav_label.setText(
            f"Label {label_id} ({index + 1}/{len(self._flagged_label_ids)}) "
            f"@ Z={z}, Y={y}, X={x}"
        )

    def _refresh_layer_options(self) -> None:
        current_image = self.image_combo.currentText()
        current_labels = self.labels_combo.currentText()

        image_names = []
        labels_names = []
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                if getattr(layer, "_epicycle_temporary", False):
                    continue  # hide transient images (guide_walls, etc.)
                ndim = layer.data.ndim
                if ndim in {2, 3}:
                    image_names.append(layer.name)
                elif ndim == 4:
                    if int(layer.data.shape[-1]) in {3, 4}:
                        image_names.append(layer.name)
            elif isinstance(layer, Labels):
                if getattr(layer, "_epicycle_temporary", False):
                    continue  # hide transient layers (Refine, preview, etc.)
                ndim = layer.data.ndim
                if ndim in {2, 3}:
                    labels_names.append(layer.name)

        self.image_combo.blockSignals(True)
        self.labels_combo.blockSignals(True)
        self.image_combo.clear()
        self.labels_combo.clear()
        self.image_combo.addItems(image_names)
        self.labels_combo.addItems(labels_names)
        self.image_combo.blockSignals(False)
        self.labels_combo.blockSignals(False)

        if current_image in image_names:
            self.image_combo.setCurrentText(current_image)
        if current_labels in labels_names:
            self.labels_combo.setCurrentText(current_labels)

        self._populate_patch_context_combo()

        if self.image_combo.count() == 0:
            self.status_label.setText("No image layers available in the main viewer.")

    def _create_labels_from_image(self) -> None:
        image_name = self.image_combo.currentText().strip()
        if not image_name:
            self.status_label.setText("Choose an image layer first.")
            return
        img = self._layer_by_name(self.viewer.layers, image_name)
        if not isinstance(img, Image):
            self.status_label.setText("Selected image layer is invalid.")
            return

        labels_name = f"{image_name}_labels"
        if self._layer_by_name(self.viewer.layers, labels_name) is not None:
            self.labels_combo.setCurrentText(labels_name)
            self.status_label.setText(f"Using existing labels layer: {labels_name}")
            return

        arr = np.asarray(img.data)
        shape = arr.shape[:3] if arr.ndim == 4 and int(arr.shape[-1]) in {3, 4} else arr.shape
        _new_lbl = self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32), name=labels_name, opacity=0.8
        )
        # brush_size locked at 1 (a product decision: brush size is 1 on every annotator
        # Labels layer; do not make it adjustable here).
        from packages.epicycle_annotator.sync import lock_brush_size_to_1
        lock_brush_size_to_1(_new_lbl)
        self._refresh_layer_options()
        self.labels_combo.setCurrentText(labels_name)
        self.status_label.setText(f"Created labels layer: {labels_name}")

    def _disconnect_rename_handlers(self) -> None:
        """Disconnect rename event handlers from currently tracked layers.

        Called before reconnecting (to prevent accumulation across reloads)
        and during teardown (to prevent stale callbacks after close).
        """
        for iname in list(getattr(self, "_active_image_layer_names", []) or []):
            try:
                layer = self.viewer.layers[iname]
                layer.events.name.disconnect(self._on_active_image_renamed)
            except (KeyError, ValueError, AttributeError):
                pass
        for lname in list(getattr(self, "_active_labels_names", []) or []):
            try:
                layer = self.viewer.layers[lname]
                layer.events.name.disconnect(self._on_active_labels_renamed)
            except (KeyError, ValueError, AttributeError):
                pass

    def _load_selected_layers(self) -> None:
        import time as _time
        _t_load0 = _time.perf_counter()
        # Disconnect rename handlers from PREVIOUS layers before reconnecting
        # to prevent callback accumulation across reloads.
        self._disconnect_rename_handlers()
        # Disconnect visibility/style handlers from old layers to prevent
        # stale handler accumulation across patch switches.
        for old_name in list(getattr(self, "_active_image_layer_names", []) or []):
            try:
                old_layer = self.viewer.layers[old_name]
                self.subviewer_manager.disconnect_layer_visibility(old_layer)
                self.subviewer_manager.disconnect_image_style(old_layer)
            except (KeyError, ValueError, AttributeError):
                pass
        for old_name in list(getattr(self, "_active_labels_names", []) or []):
            try:
                old_layer = self.viewer.layers[old_name]
                self.subviewer_manager.disconnect_layer_visibility(old_layer)
            except (KeyError, ValueError, AttributeError):
                pass

        image_name = self.image_combo.currentText().strip()
        labels_name = self.labels_combo.currentText().strip()
        image_layer = self._layer_by_name(self.viewer.layers, image_name)
        labels_layer = self._layer_by_name(self.viewer.layers, labels_name)

        if not isinstance(image_layer, Image):
            self.status_label.setText("Select a valid image layer.")
            return
        if not isinstance(labels_layer, Labels):
            self.status_label.setText("Select a valid labels layer.")
            return
        source_image_layers = self._resolve_source_image_layers(
            image_layer=image_layer,
            labels_name=labels_name,
        )
        if not source_image_layers:
            source_image_layers = [image_layer]
        patch = self._labels_patch_prefix(labels_name)
        labels_names_to_load = [labels_name]
        if str(getattr(self, "_active_target_mode", "lateral")).lower() == "axial" and patch:
            for plane in ("XZ", "YZ"):
                candidate = self._label_for_patch(patch, plane)
                if candidate and candidate not in labels_names_to_load:
                    labels_names_to_load.append(candidate)
        # Keep only valid labels present in combo/layers.
        validated_labels = []
        for lname in labels_names_to_load:
            layer = self._layer_by_name(self.viewer.layers, lname)
            if isinstance(layer, Labels):
                validated_labels.append(lname)
        if not validated_labels:
            self.status_label.setText("No valid labels context available for selected patch.")
            return

        _t_norm0 = _time.perf_counter()
        image_notes = []
        normalized_sources = []
        for src_layer in source_image_layers:
            try:
                image_volume, image_note = self._normalize_image_volume(src_layer.data)
            except Exception as exc:
                self.status_label.setText(f"Cannot load into subviewers: {exc}")
                return
            normalized_sources.append((src_layer, image_volume))
            if image_note:
                image_notes.append(image_note)
        labels_payloads = []
        labels_notes = []
        for lname in validated_labels:
            lobj = self._layer_by_name(self.viewer.layers, lname)
            if not isinstance(lobj, Labels):
                continue
            try:
                labels_volume, labels_note = self._normalize_labels_volume(
                    lobj.data,
                    normalized_sources[0][1].shape,
                    lname,
                )
            except Exception as exc:
                self.status_label.setText(f"Cannot load into subviewers: {exc}")
                return
            labels_payloads.append((lobj, labels_volume))
            if labels_note:
                labels_notes.append(labels_note)

        if not labels_payloads:
            self.status_label.setText("No labels payload available for subviewers.")
            return

        previous_labels = set(getattr(self, "_active_labels_names", []) or [])
        current_labels = {name for name, _vol in [(l.name, v) for l, v in labels_payloads]}
        for old_name in previous_labels:
            if old_name not in current_labels:
                self.sync_manager.unregister_layer(old_name)
        self._active_image_name = image_name
        self._active_image_layer_names = [str(getattr(layer, "name", "")) for layer, _ in normalized_sources]
        for src_layer, _ in normalized_sources:
            try:
                src_layer.visible = True
            except Exception:
                pass
        self._active_labels_name = labels_name
        self._active_labels_names = [str(getattr(lobj, "name", "")) for lobj, _ in labels_payloads]
        for lobj, _ in labels_payloads:
            try:
                lobj.visible = True
            except Exception:
                pass
        if patch:
            self._active_patch_id = patch
            self._set_patch_combo_to_active()
        self._set_local_work_label_visibility(patch_prefix=patch)
        self._set_main_active_layers(image_name=image_name, labels_name=labels_name)
        self._managed_layer_names = set(self._active_image_layer_names + self._active_labels_names)
        image_proxies = [self._build_image_proxy(src, vol, image_name) for src, vol in normalized_sources]
        labels_proxies = [self._build_labels_proxy(lobj, vol) for lobj, vol in labels_payloads]

        _t_norm1 = _time.perf_counter()
        print(f"[LOAD-LAYERS] normalize+prep: {_t_norm1 - _t_norm0:.3f}s")

        # Show loading indicator.
        from qtpy.QtWidgets import QApplication as _QApp
        self.status_label.setText("Loading subviewers\u2026")
        _QApp.setOverrideCursor(Qt.WaitCursor)

        import time as _time
        _t_sub0 = _time.perf_counter()
        # Resolve right-click callback from the subclass (if it exists).
        # Slow-reload path bypasses SubviewerMirrorManager.on_layer_added
        # so we need to pass the callback explicitly to keep the
        # subviewer label layers' mouse_drag_callbacks in sync with the
        # factory-driven mirror path.  See feature/right-click-navigate
        # 2026-05-10 commit message for the full wiring rationale.
        rc_factory = getattr(self, "_make_subviewer_right_click_handler", None)
        try:
            fast_ok, old_label_names = self.subviewer_manager.try_fast_reload(
                image_proxies, labels_proxies,
            )
            if fast_ok:
                # Fast path: layers were renamed + data-swapped in-place.
                # Update sync manager: unregister old label names, register new ones.
                for old_name in old_label_names:
                    self.sync_manager.unregister_layer(old_name)
                for proxy, (lobj, labels_volume) in zip(labels_proxies, labels_payloads):
                    self.sync_manager.register_shared_data(proxy.name, labels_volume)
                    self.sync_manager.connect_label_sync(lobj)
                # Fast path re-uses existing subviewer layers — if they
                # were created in a previous slow-reload that lacked
                # the right-click callback (e.g., legacy session before
                # this feature), append the callback now.  The
                # idempotent _epicycle_right_click_connected guard
                # inside add_labels_to_subviewers won't help here
                # because we're not calling that method on fast path;
                # do it directly.
                if rc_factory is not None:
                    rc_cb = None
                    try:
                        rc_cb = rc_factory()
                    except Exception:
                        rc_cb = None
                    if rc_cb is not None:
                        for proxy in labels_proxies:
                            base = proxy.name
                            for orient, vm in zip(
                                ("XY", "XZ", "YZ"),
                                (self.subviewer_manager.viewer_model_xy,
                                 self.subviewer_manager.viewer_model_xz,
                                 self.subviewer_manager.viewer_model_yz),
                            ):
                                target_name = f"{base}_{orient}"
                                for layer in vm.layers:
                                    if layer.name == target_name and not getattr(
                                        layer, "_epicycle_right_click_connected", False
                                    ):
                                        layer.mouse_drag_callbacks.append(rc_cb)
                                        layer._epicycle_right_click_connected = True
            else:
                # Slow path: full teardown + rebuild.
                self.subviewer_manager.clear()
                for i, proxy in enumerate(image_proxies):
                    self.subviewer_manager.add_image_to_subviewers(proxy)
                for i, (proxy, (lobj, _)) in enumerate(zip(labels_proxies, labels_payloads)):
                    # Resolve a fresh callback per label proxy — each
                    # invocation returns a new closure (cheap).
                    rc_cb = None
                    if rc_factory is not None:
                        try:
                            rc_cb = rc_factory()
                        except Exception:
                            rc_cb = None
                    self.subviewer_manager.add_labels_to_subviewers(
                        proxy,
                        sync_manager=self.sync_manager,
                        drawing_handler=self.drawing_handler,
                        right_click_callback=rc_cb,
                    )
        finally:
            _QApp.restoreOverrideCursor()
            _dt_sub = _time.perf_counter() - _t_sub0
            _path = "FAST" if fast_ok else "SLOW"
            print(f"[SUBVIEWER-RELOAD] {_path} path: {_dt_sub:.3f}s")

        # Sweep magicgui orphans EARLY — before the expensive draw/edit section
        # so they don't sit visible in the taskbar during property setup.
        self._sweep_magicgui_only()

        _t_post0 = _time.perf_counter()
        print(f"[LOAD-LAYERS] post-subviewer: reconnecting sync handlers...")
        # Both paths: reconnect source layer sync (guarded — safe to call always).
        for src_layer, _ in normalized_sources:
            self.subviewer_manager.sync_layer_visibility(src_layer)
            self.subviewer_manager.sync_image_style(src_layer)
        for lobj, _ in labels_payloads:
            self.subviewer_manager.sync_layer_visibility(lobj)
        print(f"[LOAD-LAYERS] sync handlers connected")

        # Connect rename hooks to real main-viewer layers so lifecycle stays stable.
        image_layer.events.name.connect(self._on_active_image_renamed)
        for lobj, _ in labels_payloads:
            lobj.events.name.connect(self._on_active_labels_renamed)
            self.sync_manager.connect_main_label_sync(lobj)
        print(f"[LOAD-LAYERS] rename+main_label_sync connected")

        # Suppress sync during batch property setup to prevent redundant
        # cascade: _apply_draw_settings sets properties on main + subviewers,
        # but without suppression the main-layer changes cascade to subviewers
        # via sync handlers (doing the same work twice).  When the main viewer
        # is ndisplay=3, this cascade triggers expensive iso_categorical renders.
        from contextlib import nullcontext
        _sync_ctx = (
            self.sync_manager.suppress_property_sync()
            if self.sync_manager is not None
            else nullcontext()
        )
        print(f"[LOAD-LAYERS] applying draw settings...")
        with _sync_ctx:
            self._apply_draw_settings()  # also calls _apply_edit_constraints internally
            self._set_subviewer_active_labels()
        print(f"[LOAD-LAYERS] draw settings done, configuring UI...")

        # Pre-warm DirectLabelColormap on all label layers (main + 3
        # subviewers) — applies + reverts a dummy colormap to force
        # napari's lazy GPU resource init now, during the load you're
        # already waiting for, instead of the user paying ~250ms per
        # subviewer (~800ms total) on their first ctrl+click multi-
        # isolation or merge dialog checkbox toggle in a session.
        # Subclasses (Curation3DAnnotator) implement _prewarm_colormap;
        # base class no-ops if not provided.
        try:
            if hasattr(self, "_prewarm_colormap"):
                self._prewarm_colormap()
        except Exception:
            pass
        self._toggle_crosshairs(bool(self.crosshairs_check.isChecked()))
        self._configure_slice_sliders()
        self._setup_channel_keybindings()
        notes = " ".join([n for n in (image_notes + labels_notes) if n]).strip()
        channel_note = ""
        all_bound = self._active_image_layer_names + self._active_labels_names
        n_bound = min(len(all_bound), len(self._CHANNEL_KEYS))
        if n_bound > 0:
            key_hint = ", ".join(
                f"{self._CHANNEL_KEYS[i]}={all_bound[i]}"
                for i in range(n_bound)
            )
            channel_note = f" Keys: {key_hint}."
        label_note = ", ".join(self._active_labels_names) if self._active_labels_names else labels_name
        base = f"Loaded native subviewers for image '{image_name}' and labels '{label_note}'.{channel_note}"
        self.status_label.setText(f"{base} {notes}".strip())
        self._refresh_overlays_all()
        # Second sweep: catch any magicgui created by the draw/edit setup.
        self._sweep_magicgui_only()
        _t_post1 = _time.perf_counter()
        print(f"[LOAD-LAYERS] post-subviewer setup: {_t_post1 - _t_post0:.3f}s")
        _dt_load = _time.perf_counter() - _t_load0
        print(f"[LOAD-LAYERS] total: {_dt_load:.3f}s")

    def _build_image_proxy(self, src_layer, image_volume, fallback_name: str = ""):
        """Build a SimpleNamespace proxy for an image source layer."""
        return SimpleNamespace(
            name=str(getattr(src_layer, "name", fallback_name)),
            data=image_volume,
            visible=bool(getattr(src_layer, "visible", True)),
            colormap=getattr(src_layer, "colormap", None),
            contrast_limits=getattr(src_layer, "contrast_limits", (0, 1)),
            opacity=float(getattr(src_layer, "opacity", 1.0)),
            blending=getattr(src_layer, "blending", "translucent"),
            gamma=float(getattr(src_layer, "gamma", 1.0)),
            events=getattr(src_layer, "events", None),
        )

    def _build_labels_proxy(self, lobj, labels_volume):
        """Build a SimpleNamespace proxy for a labels layer."""
        return SimpleNamespace(
            name=str(getattr(lobj, "name", "")),
            data=labels_volume,
            visible=bool(getattr(lobj, "visible", True)),
            opacity=float(getattr(lobj, "opacity", 1.0)),
            colormap=getattr(lobj, "colormap", None),
            blending=getattr(lobj, "blending", "translucent"),
            mode=str(getattr(lobj, "mode", "paint")),
            brush_size=int(getattr(lobj, "brush_size", 1)),
            selected_label=int(getattr(lobj, "selected_label", 1)),
            events=getattr(lobj, "events", None),
        )

    def _infer_plane_from_image_name(self, image_name: str) -> str:
        name = str(image_name or "")
        for plane in ("XY", "XZ", "YZ"):
            if name.endswith(f"_{plane}_view") or name.endswith(f"_{plane}_native_rgb_stack"):
                return plane
        return "XY"

    def _labels_patch_prefix(self, labels_name: str) -> str:
        name = str(labels_name or "")
        suffixes = [
            "_work_labels_lateral_3d",
            "_work_labels_axial_xz_3d",
            "_work_labels_axial_yz_3d",
            "_work_labels_3d",
        ]
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return ""

    def _labels_view_hint(self, labels_name: str) -> str:
        name = str(labels_name or "")
        if name.endswith("_work_labels_lateral_3d"):
            return "XY"
        if name.endswith("_work_labels_axial_xz_3d"):
            return "XZ"
        if name.endswith("_work_labels_axial_yz_3d"):
            return "YZ"
        return ""

    def _resolve_source_image_layers(self, image_layer, labels_name: str):
        if not isinstance(image_layer, Image):
            return []
        plane = self._infer_plane_from_image_name(str(getattr(image_layer, "name", "")))
        if not plane:
            plane = str(getattr(self, "_active_selected_view", "XY")).upper()
        labels_text = str(labels_name or "")
        patch_prefix = self._labels_patch_prefix(labels_text)
        if not patch_prefix:
            # No HITL naming — collect all compatible Image layers (multi-channel).
            all_images = []
            for layer in self.viewer.layers:
                if not isinstance(layer, Image):
                    continue
                if getattr(layer, "_epicycle_temporary", False):
                    continue  # hide transient images (guide_walls, etc.)
                ndim = layer.data.ndim
                if ndim in {2, 3}:
                    all_images.append(layer)
                elif ndim == 4 and int(layer.data.shape[-1]) in {3, 4}:
                    all_images.append(layer)
            return all_images if all_images else [image_layer]
        candidates = []
        for layer in self.viewer.layers:
            if not isinstance(layer, Image):
                continue
            lname = str(getattr(layer, "name", ""))
            if lname.startswith(f"{patch_prefix}_") and lname.endswith(f"_{plane}_view"):
                candidates.append(layer)
        if not candidates:
            return [image_layer]
        return sorted(candidates, key=lambda l: str(getattr(l, "name", "")))

    # Status colors/badges for patch combo (mirrors ui.py _STATUS_COLORS/_STATUS_BADGE).
    _PATCH_STATUS_COLORS = {
        "unprocessed":  "#9e9e9e",
        "predicted":    "#42a5f5",
        "committed":    "#26c6da",
        "reconstructed": "#ff9800",
        "curating":     "#e040fb",
        "accepted":     "#4caf50",
        "finalized":    "#8d6e63",
    }
    _PATCH_STATUS_BADGE = {
        "predicted":     "P ",
        "committed":     "\u2192 ",
        "reconstructed": "R ",
        "curating":      "C ",
        "accepted":      "\u2714 ",
        "finalized":     "\u25a2 ",
    }

    def _populate_patch_context_combo(self):
        from qtpy.QtGui import QColor, QBrush
        from qtpy.QtWidgets import QListView

        current = str(self.patch_context_combo.currentText() or "").strip()
        # Strip any badge prefix from current text to get the raw patch name.
        for badge in self._PATCH_STATUS_BADGE.values():
            if current.startswith(badge):
                current = current[len(badge):]
                break

        values = []
        values.extend(self._patch_ids)
        for i in range(self.labels_combo.count()):
            text = str(self.labels_combo.itemText(i))
            prefix = self._labels_patch_prefix(text)
            if prefix:
                values.append(prefix)
        seen = set()
        ordered = []
        for v in values:
            if not v or v in seen:
                continue
            seen.add(v)
            ordered.append(v)

        statuses = getattr(self, "_patch_statuses", {})
        self.patch_context_combo.blockSignals(True)
        self.patch_context_combo.clear()
        # Use item-level model for per-item colors.
        self.patch_context_combo.setView(QListView())
        for name in ordered:
            status = statuses.get(name, "")
            badge = self._PATCH_STATUS_BADGE.get(status, "")
            self.patch_context_combo.addItem(f"{badge}{name}")
            idx = self.patch_context_combo.count() - 1
            color_hex = self._PATCH_STATUS_COLORS.get(status, "")
            if color_hex:
                self.patch_context_combo.setItemData(
                    idx, QBrush(QColor(color_hex)), Qt.ForegroundRole,
                )
        self.patch_context_combo.blockSignals(False)

        target = self._active_patch_id or current
        if target:
            # Find item matching the raw patch name (with or without badge).
            for i in range(self.patch_context_combo.count()):
                text = self.patch_context_combo.itemText(i)
                if text == target or text.endswith(target):
                    self.patch_context_combo.setCurrentIndex(i)
                    break

    def _set_patch_combo_to_active(self):
        self._populate_patch_context_combo()
        if self._active_patch_id:
            for i in range(self.patch_context_combo.count()):
                text = self.patch_context_combo.itemText(i)
                if text == self._active_patch_id or text.endswith(self._active_patch_id):
                    self.patch_context_combo.setCurrentIndex(i)
                    break

    def _switch_patch_context_from_combo(self):
        """Switch to the patch selected in the dropdown.

        Delegates to the controller's canonical ``_switch_to_curation_patch``
        via the ``_patch_switch_callback``.  The controller handles guards,
        layer swap, suppression, and triggers a deferred auto-load so the
        annotator picks up the new layers.
        """
        patch_prefix = str(self.patch_context_combo.currentText() or "").strip()
        # Strip badge prefix if present.
        for badge in self._PATCH_STATUS_BADGE.values():
            if patch_prefix.startswith(badge):
                patch_prefix = patch_prefix[len(badge):]
                break
        if not patch_prefix:
            self.status_label.setText("No patch selected for switching.")
            return

        # Crash-logger breadcrumb: count switches so the reported "a couple
        # switches in" crash precondition is quantified in the session log
        # (the count appears right before the RCLICK-ENTER / faulthandler dump).
        try:
            from napari_epicycle.utils.crash_logger import breadcrumb as _bc
            self._crash_switch_count = getattr(self, "_crash_switch_count", 0) + 1
            _bc(f"PATCH-SWITCH #{self._crash_switch_count} -> {patch_prefix!r}")
        except Exception:
            pass

        cb = getattr(self, "_patch_switch_callback", None)
        if callable(cb):
            try:
                cb(patch_prefix)
            except Exception as exc:
                self.status_label.setText(f"Patch switch failed: {exc}")
                return
            return

        # Fallback: no controller callback — switch among loaded layers only.
        target_plane = str(getattr(self, "_active_selected_view", "")).upper() or self._infer_plane_from_image_name(
            self.image_combo.currentText()
        )
        label_name = self._label_for_patch(patch_prefix, target_plane)
        if not label_name:
            self.status_label.setText(f"No loaded labels context for patch '{patch_prefix}'.")
            return
        image_name = self._image_for_patch_plane(patch_prefix, target_plane)
        if not image_name:
            image_name = self._image_for_patch_plane(patch_prefix, "XY")
        if not image_name:
            self.status_label.setText(f"No loaded image context for patch '{patch_prefix}'.")
            return
        self.labels_combo.setCurrentText(label_name)
        self.image_combo.setCurrentText(image_name)
        self._active_patch_id = patch_prefix
        self._load_selected_layers()

    def _image_for_patch_plane(self, patch_prefix: str, plane: str) -> str:
        for i in range(self.image_combo.count()):
            text = str(self.image_combo.itemText(i))
            if text.startswith(f"{patch_prefix}_") and text.endswith(f"_{plane}_view"):
                return text
        for i in range(self.image_combo.count()):
            text = str(self.image_combo.itemText(i))
            if text.startswith(f"{patch_prefix}_") and text.endswith(f"_{plane}_native_rgb_stack"):
                return text
        return ""

    def _label_for_patch(self, patch_prefix: str, plane: str) -> str:
        priority = []
        p = str(plane or "").upper()
        if p == "XY":
            priority = [
                f"{patch_prefix}_work_labels_lateral_3d",
                f"{patch_prefix}_work_labels_3d",
            ]
        elif p == "XZ":
            priority = [
                f"{patch_prefix}_work_labels_axial_xz_3d",
                f"{patch_prefix}_work_labels_3d",
            ]
        elif p == "YZ":
            priority = [
                f"{patch_prefix}_work_labels_axial_yz_3d",
                f"{patch_prefix}_work_labels_3d",
            ]
        else:
            priority = [
                f"{patch_prefix}_work_labels_lateral_3d",
                f"{patch_prefix}_work_labels_axial_xz_3d",
                f"{patch_prefix}_work_labels_axial_yz_3d",
                f"{patch_prefix}_work_labels_3d",
            ]
        available = {str(self.labels_combo.itemText(i)) for i in range(self.labels_combo.count())}
        for candidate in priority:
            if candidate in available:
                return candidate
        return ""

    def _set_main_active_layers(self, *, image_name: str, labels_name: str) -> None:
        try:
            labels_layer = self._layer_by_name(self.viewer.layers, labels_name)
            if isinstance(labels_layer, Labels):
                self.viewer.layers.selection.active = labels_layer
                return
            image_layer = self._layer_by_name(self.viewer.layers, image_name)
            if isinstance(image_layer, Image):
                self.viewer.layers.selection.active = image_layer
        except Exception:
            return

    def _set_subviewer_active_labels(self) -> None:
        patch = self._labels_patch_prefix(self._active_labels_name)
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
            orient = self._orientation_for_vm(vm)
            base_name = self._active_labels_name
            if patch and target == "axial":
                if orient == "XZ":
                    base_name = self._label_for_patch(patch, "XZ") or base_name
                elif orient == "YZ":
                    base_name = self._label_for_patch(patch, "YZ") or base_name
            lname = f"{base_name}_{orient}"
            layer = self._layer_by_name(vm.layers, lname)
            if isinstance(layer, Labels):
                try:
                    vm.layers.selection.active = layer
                except Exception:
                    continue

    def _overlay_layer_name(self, orientation: str) -> str:
        return f"__epicycle_overlay_{str(orientation).upper()}"

    def _overlay_mode(self) -> str:
        text = str(getattr(self, "overlay_mode_combo", None).currentText() if hasattr(self, "overlay_mode_combo") else "Off")
        return text.strip().lower()

    def _overlay_opacity(self) -> float:
        if hasattr(self, "overlay_opacity_sb"):
            return max(0.0, min(1.0, float(self.overlay_opacity_sb.value()) / 100.0))
        return 0.55

    def _clear_overlay_for_orientation(self, orientation: str) -> None:
        vm = self._viewer_model_for_orientation(orientation)
        if vm is None:
            return
        layer = self._layer_by_name(vm.layers, self._overlay_layer_name(orientation))
        if layer is not None:
            try:
                vm.layers.remove(layer)
            except Exception:
                pass

    def _clear_all_overlays(self) -> None:
        for orient in ("XY", "XZ", "YZ"):
            self._clear_overlay_for_orientation(orient)

    def _refresh_overlays_all(self) -> None:
        for orient in ("XY", "XZ", "YZ"):
            self._refresh_overlay_for_orientation(orient)

    def _refresh_overlay_for_orientation(self, orientation: str) -> None:
        orient = str(orientation).upper()
        mode = self._overlay_mode()
        if mode in {"", "off", "none"}:
            self._clear_overlay_for_orientation(orient)
            return
        cb = getattr(self, "_overlay_slice_callback", None)
        if not callable(cb):
            self._clear_overlay_for_orientation(orient)
            return
        vm = self._viewer_model_for_orientation(orient)
        if vm is None:
            return
        slice_idx = self._safe_first_dim_index(vm)
        try:
            payload = cb(orient, slice_idx) or {}
        except TypeError:
            payload = cb(orient) or {}
        except Exception:
            payload = {}

        if not bool(payload.get("ok", False)):
            self._clear_overlay_for_orientation(orient)
            return

        overlay_data = None
        overlay_kwargs = {"opacity": self._overlay_opacity(), "blending": "additive"}
        if mode == "probability":
            overlay_data = payload.get("cellprob")
            if overlay_data is not None:
                overlay_data = np.asarray(overlay_data)
                if overlay_data.ndim == 2:
                    # Match Cellpose-review preference with a high-contrast scalar map.
                    overlay_kwargs["colormap"] = "turbo"
        elif mode == "flow":
            overlay_data = payload.get("flow")
            if overlay_data is not None:
                overlay_data = np.asarray(overlay_data)
                if overlay_data.ndim == 3 and int(overlay_data.shape[-1]) in {3, 4}:
                    overlay_kwargs["rgb"] = True
                else:
                    overlay_kwargs["colormap"] = "turbo"

        # Axial YZ payloads are cached in controller space as (Z, Y) / (Z, Y, C),
        # but the YZ pane is rendered as (Y, Z) in subviewer space.
        if overlay_data is not None and orient == "YZ":
            if getattr(overlay_data, "ndim", 0) == 2:
                overlay_data = overlay_data.T
            elif getattr(overlay_data, "ndim", 0) == 3:
                overlay_data = np.transpose(overlay_data, (1, 0, 2))

        if overlay_data is None or np.asarray(overlay_data).size == 0:
            self._clear_overlay_for_orientation(orient)
            return

        name = self._overlay_layer_name(orient)
        existing = self._layer_by_name(vm.layers, name)
        if existing is not None:
            try:
                existing.data = overlay_data
                existing.opacity = overlay_kwargs.get("opacity", existing.opacity)
                existing.blending = overlay_kwargs.get("blending", existing.blending)
                if overlay_kwargs.get("rgb", False):
                    existing.rgb = True
                if "colormap" in overlay_kwargs and hasattr(existing, "colormap"):
                    existing.colormap = overlay_kwargs["colormap"]
                existing.visible = True
                return
            except Exception:
                try:
                    vm.layers.remove(existing)
                except Exception:
                    pass
        try:
            vm.add_image(overlay_data, name=name, **overlay_kwargs)
        except Exception:
            self._clear_overlay_for_orientation(orient)

    def _set_local_work_label_visibility(self, patch_prefix: str) -> None:
        patch = str(patch_prefix or "").strip()
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        show_names = set()
        if patch:
            if target == "axial":
                show_names.add(f"{patch}_work_labels_axial_xz_3d")
                show_names.add(f"{patch}_work_labels_axial_yz_3d")
            else:
                show_names.add(f"{patch}_work_labels_lateral_3d")
            show_names.add(f"{patch}_work_labels_3d")
        for layer in self.viewer.layers:
            if not isinstance(layer, Labels):
                continue
            name = str(getattr(layer, "name", ""))
            if "_work_labels_" not in name:
                continue
            visible = name in show_names
            try:
                layer.visible = visible
            except Exception:
                pass
            try:
                layer.editable = bool(visible)
            except Exception:
                pass

    def _on_main_layer_removed(self, event) -> None:
        layer = getattr(event, "value", None)
        if layer is None:
            return
        name = str(getattr(layer, "name", "")).strip()
        if name not in self._managed_layer_names:
            return
        # During a controlled patch switch the controller removes old layers
        # before loading replacements.  Skip ALL bookkeeping and the expensive
        # subviewer teardown — _load_selected_layers will overwrite names,
        # unregister old sync, and reuse existing subviewer layers via FAST path.
        # Clearing _active_labels_name here would trigger _on_layer_inserted
        # to schedule a redundant deferred auto-load.
        if getattr(self, "_suppress_layer_removal_cleanup", False):
            self._managed_layer_names.discard(name)
            return
        self._managed_layer_names.discard(name)
        if name == self._active_image_name:
            self._active_image_name = ""
        if name == self._active_labels_name:
            self._active_labels_name = ""
            self.sync_manager.unregister_layer(name)
        self.subviewer_manager.clear()
        self._clear_all_overlays()
        self._toggle_crosshairs(False)
        self._configure_slice_sliders()
        self.status_label.setText("Managed layer removed from viewer. Reload image/labels into subviewers.")

    def _on_active_image_renamed(self, event) -> None:
        layer = getattr(event, "source", None)
        if layer is None:
            return
        new_name = str(getattr(layer, "name", "")).strip()
        if not new_name:
            return
        self._managed_layer_names.discard(self._active_image_name)
        self._active_image_name = new_name
        self._managed_layer_names.add(new_name)
        self._refresh_layer_options()

    def _on_active_labels_renamed(self, event) -> None:
        layer = getattr(event, "source", None)
        if layer is None:
            return
        new_name = str(getattr(layer, "name", "")).strip()
        if not new_name:
            return
        old_name = self._active_labels_name
        self._managed_layer_names.discard(old_name)
        self._active_labels_name = new_name
        self._managed_layer_names.add(new_name)
        if old_name and old_name != new_name:
            self.sync_manager.rename_layer(old_name, new_name)
        self._refresh_layer_options()

    def _all_subviewer_label_layers(self):
        names = list(getattr(self, "_active_labels_names", []) or [])
        if not names and self._active_labels_name:
            names = [self._active_labels_name]
        if not names:
            return []
        layers = []
        for base_name in names:
            for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
                layer = self._layer_by_name(
                    vm.layers, f"{base_name}_{self._orientation_for_vm(vm)}"
                )
                if isinstance(layer, Labels):
                    layers.append(layer)
        return layers

    def _apply_edit_constraints(self) -> None:
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        for layer in self._all_subviewer_label_layers():
            name = str(getattr(layer, "name", ""))
            base, orient = name.rsplit("_", 1)
            orient = orient.upper()
            cls = "legacy"
            if "_work_labels_lateral_3d" in base:
                cls = "lateral"
            elif "_work_labels_axial_xz_3d" in base:
                cls = "axial_xz"
            elif "_work_labels_axial_yz_3d" in base:
                cls = "axial_yz"

            if target == "3d_curation":
                # In 3D curation mode, all views are editable
                allow = True
            elif target == "lateral":
                allow = orient == "XY" and cls in {"lateral", "legacy"}
            else:
                allow = (orient == "XZ" and cls in {"axial_xz", "legacy"}) or (
                    orient == "YZ" and cls in {"axial_yz", "legacy"}
                )
            try:
                # Journey #14 (2026-09-07): only the DISALLOWED branch may
                # touch visibility.  Forcing `visible = allow` on both
                # branches un-hid a layer the user had deliberately hidden
                # with the `d` channel keybinding on the next refresh (e.g.
                # after a prediction).  A hidden layer stays non-editable —
                # napari swallows the paint keybinding on invisible layers
                # anyway.
                if not allow:
                    layer.visible = False
                # `editable` follows the TARGET MODE only, never the layer's
                # current visibility: nothing re-arms it when the layer becomes
                # visible again, so `editable = allow and visible` latched a
                # layer hidden with `d` into permanently non-editable (and, in
                # Step 3, downgraded layers that are transiently invisible
                # under the isolation cascade — the parked
                # paint-after-subviewer-switch bug).  napari already forces
                # PAN_ZOOM while a layer is invisible (base.py:748-749) and
                # restores the mode on re-show.
                layer.editable = bool(allow)
                if not allow:
                    layer.mode = "pan_zoom"
            except Exception:
                pass

    def _orientation_for_vm(self, vm) -> str:
        if vm is self.viewer_model_xy:
            return "XY"
        if vm is self.viewer_model_xz:
            return "XZ"
        return "YZ"

    def _apply_draw_settings(self) -> None:
        mode = str(getattr(self, "_default_draw_mode", "paint")).strip().lower() or "paint"
        # brush_size hardcoded: it is fixed at 1 on every annotator Labels layer
        # (a product decision; do not make it adjustable here).
        # _default_brush_size remains on the instance for backwards
        # compatibility but its value is ignored; lock_brush_size_to_1 +
        # the snap-back handler are the source of truth.
        from packages.epicycle_annotator.sync import lock_brush_size_to_1, _BRUSH_LOCKED_VALUE
        brush_size = _BRUSH_LOCKED_VALUE
        enclosed_fill = bool(getattr(self, "_enclosed_fill_enabled", True))
        auto_increment = bool(getattr(self, "_auto_increment_labels", True))
        selected_label = int(getattr(self, "_default_selected_label", 1))

        main_labels = self._layer_by_name(self.viewer.layers, self._active_labels_name)
        _ndisplay = getattr(
            getattr(self.viewer, "dims", None), "ndisplay", 2
        )
        if isinstance(main_labels, Labels):
            existing = int(getattr(main_labels, "selected_label", selected_label))
            selected_label = max(1, existing)
            # Lock + seed even in 3D — the lock is a single int compare
            # in steady state, and ensures the slider can't drift if a
            # session was started before this fix landed.
            lock_brush_size_to_1(main_labels)
            if _ndisplay < 3:
                # 2D: safe to set all properties — no expensive 3D renders.
                main_labels.mode = mode
                main_labels.selected_label = selected_label
            else:
                # 3D: skip mode/selected_label changes on the main
                # iso_categorical layer.  Any property change can trigger
                # expensive isosurface re-renders or cascade through the
                # sync manager.  Painting happens on subviewers (2D), so
                # the main layer doesn't need these.  It already retains
                # correct values from the previous session or from
                # set_context().  brush_size lock is excluded from this
                # skip because lock_brush_size_to_1 is no-op when already
                # at 1.
                pass
            setattr(main_labels, "_epicycle_enclosed_fill_enabled", enclosed_fill)
            setattr(main_labels, "_epicycle_auto_increment_label", auto_increment)

        for layer in self._all_subviewer_label_layers():
            layer.mode = mode
            lock_brush_size_to_1(layer)
            layer.selected_label = max(1, int(getattr(layer, "selected_label", selected_label)))
            setattr(layer, "_epicycle_enclosed_fill_enabled", enclosed_fill)
            setattr(layer, "_epicycle_auto_increment_label", auto_increment)
        self._apply_edit_constraints()
        self.status_label.setText(
            "Quick annotate defaults active: paint + close/fill + auto label increment (brush=1)."
        )

    def _toggle_crosshairs(self, enabled: bool) -> None:
        try:
            self.crosshair_manager.toggle_crosshairs(bool(enabled))
            if enabled:
                self.crosshair_manager.update_crosshairs()
        except Exception as exc:
            self.status_label.setText(f"Crosshairs unavailable: {exc}")
            return
        if enabled:
            self.status_label.setText("Crosshairs enabled.")
        else:
            self.status_label.setText("Crosshairs disabled.")

    # ---------------------------------------------------------------
    # Channel visibility keybindings (a / s / d / f)
    # ---------------------------------------------------------------

    _CHANNEL_KEYS = ("a", "s", "d", "f")

    def _setup_channel_keybindings(self) -> None:
        """Bind a/s/d/f keys to toggle visibility of loaded layers.

        Keys are mapped positionally: image channels first, then labels layers.
        For example with 2 image channels and 1 labels layer:
        a → 1st image, s → 2nd image, d → labels layer.
        Bindings are registered on all viewers (main + 3 subviewers)
        so the keys work regardless of which panel has focus.
        """
        self._teardown_channel_keybindings()
        image_names = list(getattr(self, "_active_image_layer_names", []) or [])
        labels_names = list(getattr(self, "_active_labels_names", []) or [])
        # Combined list: image channels first, then labels layers
        all_layer_names = image_names + labels_names
        if not all_layer_names:
            return

        viewers = [
            self.viewer,
            self.viewer_model_xy,
            self.viewer_model_xz,
            self.viewer_model_yz,
        ]
        bound: list[tuple] = []  # (viewer, key) pairs for teardown

        for idx, key in enumerate(self._CHANNEL_KEYS):
            if idx >= len(all_layer_names):
                break
            layer_name = all_layer_names[idx]

            def _make_toggle(name: str):
                """Create a closure that toggles visibility of *name* across all viewers."""
                def _toggle(_viewer):
                    # FM8 fix (2026-05-16): route through the
                    # `_get_toggle_targets_for_name` hook so subclasses
                    # (Curation3DAnnotator) can swap in their isolation
                    # proxy / Refine layers when those modes are active.
                    # Default impl returns main labels + subviewer labels.
                    # Reentrancy guard — users mash hide/show
                    # keys rapidly.  Without this, a second keypress
                    # could enter _toggle mid-cascade and corrupt the
                    # saved pre-hide mode.  Drop policy: one of N rapid
                    # presses lands.  Canary print stays so any real-
                    # world reentry shows up in user logs.
                    if getattr(self, "_channel_toggle_in_flight", False):
                        print(f"[§2.5] reentrant toggle DROPPED name={name!r} key={key!r}")
                        return
                    self._channel_toggle_in_flight = True
                    try:
                        targets = self._get_toggle_targets_for_name(name)
                        if not targets:
                            return

                        # 2026-05-23 (v2 fix): directional save/
                        # restore.  Without this, the SUBVIEWER MIRRORS
                        # get their ``_visible_mode`` corrupted during
                        # hide: ``sync_label_modes`` propagates main's
                        # pan_zoom to mirrors BEFORE each mirror's
                        # ``_on_visible_changed`` saves its own mode,
                        # so the mirror's _visible_mode captures the
                        # corrupted value.  On show napari restores
                        # main correctly but each mirror restores from
                        # its corrupted save, undoing
                        # ``sync_label_modes``' subsequent
                        # re-propagation of the restored main mode.
                        # Workaround: SAVE modes BEFORE the visibility
                        # toggle (uncorrupted), RESTORE AFTER the
                        # toggle from a persistent per-layer attribute.
                        # The editable gate on RESTORE preserves layers
                        # that are intentionally non-editable (e.g.
                        # XZ/YZ mirrors in Step 2 lateral mode).  Full
                        # trace in commits a922e48 + e6cb7ca.
                        new_vis = not bool(getattr(targets[0], "visible", True))

                        if not new_vis:
                            for lyr in targets:
                                try:
                                    lyr._epicycle_pre_hide_mode = str(lyr.mode)
                                except Exception:
                                    pass

                        for lyr in targets:
                            try:
                                lyr.visible = new_vis
                            except Exception:
                                pass

                        if new_vis:
                            for lyr in targets:
                                try:
                                    saved = getattr(lyr, "_epicycle_pre_hide_mode", None)
                                    if saved and getattr(lyr, "editable", False) and hasattr(lyr, "mode"):
                                        lyr.mode = saved
                                except Exception:
                                    pass
                    finally:
                        self._channel_toggle_in_flight = False
                return _toggle

            toggle_fn = _make_toggle(layer_name)
            for v in viewers:
                try:
                    v.bind_key(key, toggle_fn, overwrite=True)
                    bound.append((v, key))
                except Exception:
                    pass

        self._channel_keybindings = bound

    def _get_toggle_targets_for_name(self, name: str) -> list:
        """Return the list of layers the channel-aware visibility key
        (`_CHANNEL_KEYS`) should toggle for the given logical layer
        name.

        Default impl (this base class): main labels layer + one
        subviewer label layer per orientation (name_XY / name_XZ /
        name_YZ, or the same name if no orient suffix).  Preserves
        the earlier behavior for callers that don't override.

        Subclass override point — `Curation3DAnnotator` routes around
        isolation proxies and the Refine layer when those modes are
        active, since the main labels layer is intentionally hidden
        by the isolation system in those states.
        """
        targets: list = []
        main_layer = self._layer_by_name(self.viewer.layers, name)
        if main_layer is not None:
            targets.append(main_layer)
        for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
            orient = self._orientation_for_vm(vm)
            sub_layer = self._layer_by_name(vm.layers, f"{name}_{orient}")
            if sub_layer is None:
                sub_layer = self._layer_by_name(vm.layers, name)
            if sub_layer is not None:
                targets.append(sub_layer)
        return targets

    def _teardown_channel_keybindings(self) -> None:
        """Remove previously bound channel visibility keys."""
        for viewer, key in getattr(self, "_channel_keybindings", []):
            try:
                viewer.bind_key(key, None, overwrite=True)
            except Exception:
                pass
        self._channel_keybindings = []

    def _toggle_advanced_data(self, enabled: bool) -> None:
        if hasattr(self, "data_group") and self.data_group is not None:
            self.data_group.setVisible(bool(enabled))
        btn = getattr(self, "advanced_data_btn", None)
        if btn is None:
            return
        if bool(enabled):
            btn.setText("Hide Advanced")
        else:
            btn.setText("Advanced Data")

    def _request_segment_slice(self, orientation: str) -> None:
        orient = str(orientation or "").upper()
        if orient not in {"XY", "XZ", "YZ"}:
            self.status_label.setText(f"Unsupported orientation: {orientation}")
            return
        vm = self._viewer_model_for_orientation(orient)
        requested_slice = self._safe_first_dim_index(vm) if vm is not None else 0
        cb = getattr(self, "_segment_slice_callback", None)
        if not callable(cb):
            self.status_label.setText("Segmentation callback is not available.")
            return
        try:
            try:
                result = cb(orient, slice_index=requested_slice) or {}
            except TypeError:
                # Backward compatibility with legacy callback signature.
                result = cb(orient) or {}
        except Exception as exc:
            self.status_label.setText(f"Segmentation failed for {orient}: {exc}")
            return

        ok = bool(result.get("ok", False))
        message = str(result.get("message", "")).strip()
        image_name = str(result.get("image_layer_name", "")).strip()
        labels_name = str(result.get("labels_layer_name", "")).strip()
        if result.get("active_patch_id"):
            self._active_patch_id = str(result.get("active_patch_id"))
            self._set_patch_combo_to_active()
        if image_name and self.image_combo.findText(image_name) >= 0:
            self.image_combo.setCurrentText(image_name)
        if labels_name and self.labels_combo.findText(labels_name) >= 0:
            self.labels_combo.setCurrentText(labels_name)
        saved_steps = self._capture_subviewer_steps()
        target_slice = int(result.get("slice_index", requested_slice))
        if self.image_combo.currentText() and self.labels_combo.currentText():
            self._refresh_label_data_in_subviewers()
            saved_steps[orient] = target_slice
            self._restore_subviewer_steps(saved_steps)
            self._refresh_overlays_all()
        if message:
            self.status_label.setText(message)
        else:
            self.status_label.setText(
                f"{'Segmented' if ok else 'Unable to segment'} {orient} slice."
            )

    def _request_segment_active_mode(self) -> None:
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        if target == "axial":
            preferred = str(getattr(self, "_active_selected_view", "XZ")).upper()
            order = [preferred] if preferred in {"XZ", "YZ"} else ["XZ"]
            for plane in ["XZ", "YZ"]:
                if plane not in order:
                    order.append(plane)
            results = []
            for plane in order:
                vm = self._viewer_model_for_orientation(plane)
                requested_slice = self._safe_first_dim_index(vm) if vm is not None else 0
                cb = getattr(self, "_segment_slice_callback", None)
                if not callable(cb):
                    self.status_label.setText("Segmentation callback is not available.")
                    return
                try:
                    try:
                        result = cb(plane, slice_index=requested_slice) or {}
                    except TypeError:
                        result = cb(plane) or {}
                except Exception as exc:
                    result = {"ok": False, "message": f"Segmentation failed for {plane}: {exc}"}
                results.append(result)
                image_name = str(result.get("image_layer_name", "")).strip()
                labels_name = str(result.get("labels_layer_name", "")).strip()
                if image_name and self.image_combo.findText(image_name) >= 0:
                    self.image_combo.setCurrentText(image_name)
                if labels_name and self.labels_combo.findText(labels_name) >= 0:
                    self.labels_combo.setCurrentText(labels_name)
            saved_steps = self._capture_subviewer_steps()
            for plane, result in zip(order, results):
                if result:
                    saved_steps[plane] = int(result.get("slice_index", saved_steps.get(plane, 0)))
            if self.image_combo.currentText() and self.labels_combo.currentText():
                self._refresh_label_data_in_subviewers()
                self._restore_subviewer_steps(saved_steps)
                self._refresh_overlays_all()
            ok_count = sum(1 for r in results if bool(r.get("ok", False)))
            total = len(results)
            if ok_count == total:
                self.status_label.setText("Segmented axial slices (XZ + YZ).")
            elif ok_count > 0:
                self.status_label.setText("Partially segmented axial slices (one plane failed).")
            else:
                self.status_label.setText("Axial segmentation failed for both XZ and YZ.")
            return

        self._request_segment_slice("XY")

    def _request_accept_active_slice(self) -> None:
        """Accept the currently visible slice(s) in the active orientation.

        In axial mode, accepts both XZ and YZ (mirroring segment behavior).
        Empty-mask slices are silently skipped by the controller callback.
        """
        cb = getattr(self, "_accept_slice_callback", None)
        if not callable(cb):
            self.status_label.setText("Accept callback is not available.")
            return

        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        if target == "axial":
            # Accept both XZ and YZ, same as segment does.
            accepted = []
            skipped = []
            for plane in ("XZ", "YZ"):
                vm = self._viewer_model_for_orientation(plane)
                slice_idx = self._safe_first_dim_index(vm) if vm is not None else 0
                try:
                    result = cb(plane, slice_index=slice_idx) or {}
                except Exception as exc:
                    skipped.append(f"{plane}: {exc}")
                    continue
                if result.get("ok"):
                    accepted.append(plane)
                else:
                    skipped.append(f"{plane}: {result.get('message', 'failed')}")

            if accepted:
                self.status_label.setText(f"Accepted axial slices ({' + '.join(accepted)}).")
            elif skipped:
                self.status_label.setText(f"No axial slices accepted: {'; '.join(skipped)}")
            else:
                self.status_label.setText("No axial slices to accept.")
        else:
            vm = self._viewer_model_for_orientation("XY")
            slice_idx = self._safe_first_dim_index(vm) if vm is not None else 0
            try:
                result = cb("XY", slice_index=slice_idx) or {}
            except Exception as exc:
                self.status_label.setText(f"Accept failed for XY: {exc}")
                return

            ok = bool(result.get("ok", False))
            message = str(result.get("message", "")).strip()
            if message:
                self.status_label.setText(message)
            else:
                self.status_label.setText(
                    f"{'Accepted' if ok else 'Unable to accept'} XY slice."
                )

    def _refresh_label_data_in_subviewers(self) -> None:
        """Lightweight refresh: update label data in existing subviewer layers.

        Unlike ``_load_selected_layers`` which tears down and rebuilds all layers,
        this just refreshes the label arrays in-place.  Much faster after
        segmentation where only the mask data has changed.

        Phase A #2 (perf push 2026-05-09):
          - Identity check before the data setter — skips napari's setter
            cascade (and the ``events.data`` emit) when the transposed
            view shares object identity with ``sub_layer.data``.  Safe
            because in that case ``shared_labels[name]`` (the sync.py
            listener target) already points at the same object — no
            drift possible.
          - Subviewers in 3D mode get a deferred refresh via
            ``QTimer.singleShot(50, _safe_deferred_refresh(sub_layer))``
            instead of synchronous refresh().  In 3D mode, refresh()
            triggers an iso_categorical GPU recompute against the full
            label texture (1–2 s at 1024³); deferring lets the calling
            code (refinement op accept) return immediately and the GPU
            work fires on next event-loop tick.  The wrapper is
            sip-deletion-safe — see ``_safe_deferred_refresh`` above.
          - 2D-mode subviewers keep synchronous refresh (2D is cheap).

        See commit message for full hydra-check trace.
        """
        from packages.epicycle_annotator.orientation import transpose_data

        for lname in (getattr(self, "_active_labels_names", None) or [self._active_labels_name]):
            if not lname:
                continue
            main_layer = self._layer_by_name(self.viewer.layers, lname)
            if not isinstance(main_layer, Labels):
                continue
            data = main_layer.data
            for orient, vm in [
                ("XY", self.viewer_model_xy),
                ("XZ", self.viewer_model_xz),
                ("YZ", self.viewer_model_yz),
            ]:
                sub_name = f"{lname}_{orient}"
                sub_layer = self._layer_by_name(vm.layers, sub_name)
                if sub_layer is None:
                    continue
                try:
                    transposed = transpose_data(data, orient)
                    if transposed is not sub_layer.data:
                        sub_layer.data = transposed
                    ndisplay = getattr(getattr(vm, "dims", None), "ndisplay", 2)
                    if ndisplay == 3:
                        # Capture sub_layer in default arg to avoid
                        # late-binding bugs if the loop variable rebinds
                        # before the timer fires.
                        QTimer.singleShot(
                            50,
                            lambda layer=sub_layer: _safe_deferred_refresh(layer),
                        )
                    else:
                        sub_layer.refresh()
                except Exception:
                    pass

        # 2026-05-23 fix — re-assert mode-aware editability constraints.
        #
        # napari's ``Labels._reset_editable`` (labels.py:814) runs as a
        # side effect of ``layer.data = ...`` (labels.py:586) and forces
        # ``editable=True`` on the layer regardless of our constraint
        # state.  Without this re-application, after any
        # ``_refresh_label_data_in_subviewers`` call (e.g. post-Cellpose-
        # prediction via ``_request_segment_slice`` →
        # ``_refresh_label_data_in_subviewers``) the XZ and YZ subviewer
        # mirrors flip from ``editable=False`` (correct for lateral mode)
        # back to ``editable=True``.  This in turn lets ``_do_pick_refine``
        # set ``mode='paint'`` on them via the ``if sub_layer.editable:``
        # gate, ultimately allowing the user to paint on axial subviewers
        # in lateral mode.
        #
        # Trace evidence (caller stack from [DIAG-WATCH] on 2026-05-23):
        #   labels.py:814(_reset_editable)
        #     <- labels.py:586(data)
        #     <- native_multiview_widget.py:2583(this function)
        #     <- native_multiview_widget.py:2422(_request_segment_slice)
        #     <- native_multiview_widget.py:2481(_request_segment_active_mode)
        #
        # ``_apply_edit_constraints`` is idempotent and reads
        # ``self._active_target_mode``, so it correctly locks down
        # XZ/YZ in lateral mode and locks down XY in axial mode.
        # No-op when in 3D-curation mode (Step 3) since ``allow=True``
        # for all orientations there.
        try:
            self._apply_edit_constraints()
        except Exception:
            pass

    def navigate_to_slice(self, orientation: str, slice_index: int) -> None:
        """Navigate the subviewer for the given orientation to the specified slice index."""
        orient = str(orientation).upper()
        vm = self._viewer_model_for_orientation(orient)
        if vm is None:
            return
        slider = self._slice_slider_for_orientation(orient)
        try:
            vm.dims.set_current_step(0, int(slice_index))
        except Exception:
            pass
        if slider is not None:
            try:
                value = max(slider.minimum(), min(slider.maximum(), int(slice_index)))
                self._syncing_slice_slider[orient] = True
                slider.setValue(value)
                self._syncing_slice_slider[orient] = False
            except Exception:
                pass
        try:
            self.crosshair_manager.update_crosshairs()
        except Exception:
            pass
        self._refresh_overlay_for_orientation(orient)

    def _refresh_segment_button_text(self) -> None:
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        if target == "axial":
            self.segment_mode_btn.setText("Segment Axial (XZ + YZ)")
            self.accept_slice_btn.setText("Accept Axial (XZ + YZ)")
        else:
            self.segment_mode_btn.setText("Segment Lateral (XY)")
            self.accept_slice_btn.setText("Accept Lateral (XY)")

    def _safe_first_dim_index(self, viewer_model) -> int:
        try:
            step = tuple(getattr(viewer_model.dims, "current_step", ()))
            if len(step) > 0:
                return int(step[0])
        except Exception:
            pass
        return 0

    def _open_advanced_features(self) -> None:
        """Open the Advanced Features dock widget."""
        try:
            from napari_epicycle.annotator.launcher import launch_advanced_features_dock
            launch_advanced_features_dock(self.viewer)
        except Exception:
            pass

    def _open_multiviewer_settings(self) -> None:
        """Open the Multiviewer Settings dock widget."""
        try:
            from napari_epicycle.annotator.launcher import (
                launch_multiviewer_settings_dialog,
            )
            launch_multiviewer_settings_dialog(self.viewer)
        except Exception:
            pass

    def _viewer_model_for_orientation(self, orientation: str):
        o = str(orientation).upper()
        if o == "XY":
            return self.viewer_model_xy
        if o == "XZ":
            return self.viewer_model_xz
        return self.viewer_model_yz

    def _slice_slider_for_orientation(self, orientation: str):
        o = str(orientation).upper()
        if o == "XY":
            return getattr(self, "xy_slice_slider", None)
        if o == "XZ":
            return getattr(self, "xz_slice_slider", None)
        return getattr(self, "yz_slice_slider", None)

    def _configure_slice_sliders(self) -> None:
        for orient in ["XY", "XZ", "YZ"]:
            self._configure_single_slice_slider(orient)

    def _configure_single_slice_slider(self, orientation: str) -> None:
        slider = self._slice_slider_for_orientation(orientation)
        vm = self._viewer_model_for_orientation(orientation)
        if slider is None or vm is None:
            return
        max_idx = 0
        if self._active_image_name:
            img = self._layer_by_name(vm.layers, f"{self._active_image_name}_{orientation}")
            if img is not None:
                try:
                    if img.data.ndim >= 1:
                        max_idx = max(0, int(img.data.shape[0]) - 1)
                except Exception:
                    max_idx = 0
        self._syncing_slice_slider[orientation] = True
        try:
            slider.setRange(0, max_idx)
            slider.setValue(min(self._safe_first_dim_index(vm), max_idx))
            slider.setVisible(max_idx > 0)
        finally:
            self._syncing_slice_slider[orientation] = False

    def _on_slice_slider_changed(self, orientation: str, value: int) -> None:
        o = str(orientation).upper()
        if self._syncing_slice_slider.get(o, False):
            return
        vm = self._viewer_model_for_orientation(o)
        if vm is None:
            return
        try:
            vm.dims.set_current_step(0, int(value))
            self.crosshair_manager.update_crosshairs()
            self._refresh_overlay_for_orientation(o)
        except Exception:
            return

    def _sync_slider_from_dims(self, orientation: str) -> None:
        o = str(orientation).upper()
        slider = self._slice_slider_for_orientation(o)
        vm = self._viewer_model_for_orientation(o)
        if slider is None or vm is None:
            return
        if self._syncing_slice_slider.get(o, False):
            return
        self._syncing_slice_slider[o] = True
        try:
            idx = self._safe_first_dim_index(vm)
            idx = max(slider.minimum(), min(slider.maximum(), idx))
            slider.setValue(idx)
            self._refresh_overlay_for_orientation(o)
        finally:
            self._syncing_slice_slider[o] = False

    def _capture_subviewer_steps(self) -> dict:
        return {
            "XY": self._safe_first_dim_index(self.viewer_model_xy),
            "XZ": self._safe_first_dim_index(self.viewer_model_xz),
            "YZ": self._safe_first_dim_index(self.viewer_model_yz),
        }

    def _restore_subviewer_steps(self, steps: dict) -> None:
        for orient in ("XY", "XZ", "YZ"):
            vm = self._viewer_model_for_orientation(orient)
            if vm is None:
                continue
            value = int(steps.get(orient, 0))
            try:
                vm.dims.set_current_step(0, value)
            except Exception:
                continue
        try:
            self.crosshair_manager.update_crosshairs()
        except Exception:
            pass
        self._refresh_overlays_all()
