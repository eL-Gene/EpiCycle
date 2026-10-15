"""Curation2DAnnotator — dedicated 2D HITL annotator subclass.

Extends NativeMultiviewAnnotator with proper lifecycle management for
the HITL 2D predict-curate loop.  Mirrors the pattern established by
Curation3DAnnotator for 3D curation:

- Deferred auto-load via QTimer (prevents double-load on launch)
- Session token gating (prevents deferred callbacks firing into dead widgets)
- Full _close_cleanup with handler disconnection and callback nulling
- closeEvent override with subviewer cleanup and orphan sweep

The base class's Quick Annotate controls (Segment, Accept, Overlay)
are kept visible — they ARE the core 2D HITL loop.
"""

from __future__ import annotations

import time as _time
import traceback as _tb
from pathlib import Path as _Path

import numpy as np
from napari.layers import Labels
from qtpy.QtCore import Qt, QEvent, QObject, QRect, QTimer
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from .native_multiview_widget import NativeMultiviewAnnotator
from ..ui_shell.theme import help_button as _help_button


# 2026-06-01: per-section help buttons (?) launching themed_help_dialog.
# Guides live under docs/guides/ (an installed wheel: napari_epicycle/guides/);
# ui_shell._guides.guides_dir() picks the folder.
from napari_epicycle.ui_shell._guides import guides_dir as _guides_dir
_ANNOTATOR_GUIDES_DIR = _guides_dir()


# ---------------------------------------------------------------------------
# Quick Annotate panel styling (2026-05-27 aesthetic refresh)
# ---------------------------------------------------------------------------
#
# Color convention: match the existing crosshair colors so users get a
# consistent visual identity per orientation across the whole annotator.
# Source: native_multiview_widget.py:298-300 (crosshair color dict).
#
# Per-view buttons (Segment XY/XZ/YZ, Accept XY/XZ/YZ) have 3 states:
#   - At rest (enabled, not the active paint surface): transparent fill,
#     thin colored left edge in the orient tint.
#   - Active paint surface (focused subviewer + paint mode): button
#     background fills with the orient color at low opacity (~25%),
#     border becomes the orient color.  No separate "amber accent" — the
#     orient color IS the active-state signal.  Cyan-active XZ, magenta-
#     active YZ, yellow-active XY → instantly identifiable.
#   - Disabled (wrong bucket): Qt's native dimmed state.
#
# Non-orient buttons (Switch, Advanced Features..., Multiviewer Settings...)
# use _NEUTRAL_BTN_STYLE — thin border, transparent fill, subtle hover.
#
# Dropdowns (Patch combo, Overlay combo) use _DROPDOWN_STYLE — thin border,
# dark fill, no heavy default-napari gray.

_ORIENT_TINT = {
    "XY": "#f3e55b",  # yellow (matches crosshair)
    "XZ": "#52e8ff",  # cyan
    "YZ": "#ff4fd8",  # magenta
}
_ORIENT_TINT_RGB = {
    "XY": (243, 229, 91),
    "XZ": (82, 232, 255),
    "YZ": (255, 79, 216),
}


# Edit-state accents (2026-05-27 edit-states branch): when the displayed
# slice for an orientation is finalized, its Segment/Accept buttons switch
# from the per-orient tint to an edit-state accent so the user sees at a
# glance that the slice is locked-in.  emerald = accepted, ice = staged/
# committed (matches the main-widget status icons: accept=#6abf69 green,
# staged-diamond=#80deea ice).
_EDIT_STATE_ACCENT = {
    "accepted": ("#2ecc71", (46, 204, 113)),    # emerald
    "finalized": ("#80deea", (128, 222, 234)),  # ice (staged + committed)
}


def _orient_button_style(orient: str, paint_active: bool = False,
                         edit_state: str = "editable") -> str:
    """QSS for per-view Segment/Accept buttons in Quick Annotate panel.

    edit_state (2026-05-27): "editable" keeps the normal per-orient tint;
    "accepted" / "finalized" paint an emerald / ice locked-look accent so
    the user can see the displayed slice is finalized.  The buttons are
    typically ALSO disabled by the caller in those states (soft lock).
    """
    if edit_state in _EDIT_STATE_ACCENT:
        accent, (r, g, b) = _EDIT_STATE_ACCENT[edit_state]
        return f"""
            QPushButton {{
                background-color: rgba({r}, {g}, {b}, 20);
                border: 1px solid rgba({r}, {g}, {b}, 90);
                border-left: 5px solid {accent};
                padding: 5px 8px;
                color: #cfcfcf;
                border-radius: 3px;
            }}
            QPushButton:disabled {{
                color: #888;
                background-color: rgba({r}, {g}, {b}, 12);
                border: 1px solid rgba({r}, {g}, {b}, 55);
                border-left: 5px solid {accent};
                font-weight: normal;
            }}
        """
    tint = _ORIENT_TINT.get(str(orient).upper(), "#888888")
    r, g, b = _ORIENT_TINT_RGB.get(str(orient).upper(), (136, 136, 136))
    if paint_active:
        # Active state: button background IS the orient color at ~25% opacity.
        # No separate amber accent — the color itself communicates which view
        # is the active paint surface.
        return f"""
            QPushButton {{
                background-color: rgba({r}, {g}, {b}, 65);
                border: 1px solid {tint};
                border-left: 5px solid {tint};
                padding: 5px 8px;
                color: #fff;
                font-weight: bold;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: rgba({r}, {g}, {b}, 95);
            }}
            QPushButton:disabled {{
                color: #666;
                background: transparent;
                border: 1px solid #333;
                border-left: 3px solid #444;
                font-weight: normal;
            }}
        """
    return f"""
        QPushButton {{
            background-color: transparent;
            border: 1px solid #555;
            border-left: 4px solid {tint};
            padding: 5px 8px;
            color: #ddd;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            border-color: {tint};
            background-color: rgba({r}, {g}, {b}, 28);
        }}
        QPushButton:pressed {{
            background-color: rgba({r}, {g}, {b}, 50);
        }}
        QPushButton:disabled {{
            color: #666;
            background: transparent;
            border: 1px solid #333;
            border-left: 3px solid #444;
        }}
    """


# Neutral utility button — Switch, Advanced Features..., Multiviewer Settings...
_NEUTRAL_BTN_STYLE = """
    QPushButton {
        background-color: transparent;
        border: 1px solid #555;
        padding: 5px 10px;
        color: #ddd;
        border-radius: 3px;
    }
    QPushButton:hover {
        border-color: #888;
        background-color: rgba(255, 255, 255, 14);
    }
    QPushButton:pressed {
        background-color: rgba(255, 255, 255, 24);
    }
    QPushButton:disabled {
        color: #666;
        border-color: #333;
    }
"""

# Dropdown — Patch combo, Overlay mode combo.
_DROPDOWN_STYLE = """
    QComboBox {
        background-color: #2a2a2a;
        border: 1px solid #555;
        border-radius: 3px;
        padding: 4px 8px;
        color: #ddd;
        min-height: 18px;
    }
    QComboBox:hover {
        border-color: #777;
    }
    QComboBox:focus {
        border-color: #999;
    }
    QComboBox:disabled {
        color: #666;
        border-color: #333;
        background-color: #1f1f1f;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
        subcontrol-position: right center;
    }
    QComboBox QAbstractItemView {
        background-color: #2a2a2a;
        border: 1px solid #555;
        color: #ddd;
        selection-background-color: #444;
        selection-color: #fff;
        outline: 0;
    }
"""

# Section label — small, dim, used as group headers in the panel.
_SECTION_LABEL_STYLE = (
    "color: #888888; font-size: 10px; font-weight: 600; "
    "letter-spacing: 0.5px; text-transform: uppercase; "
    "padding: 4px 0px 2px 0px;"
)

# Horizontal separator — thin line between panel sections.
_SEP_LINE_STYLE = "background-color: #3a3a3a; max-height: 1px; margin: 2px 0px;"


class _SubviewerFocusFilter(QObject):
    """Qt event filter: detects mouse press on a subviewer canvas and
    notifies the annotator which orientation was focused.

    Installed on qt_viewer.canvas.native (same target as _CurationKeyFilter).
    Does not consume events — just observes and forwards.
    """

    def __init__(self, annotator, orient, parent=None):
        super().__init__(parent)
        self._annotator = annotator
        self._orient = orient

    def eventFilter(self, obj, event):
        # C1 (review follow-up): only fire on LEFT-button press.  Right-click
        # context menus and middle-click pans were also triggering layer
        # reassignment under the generic MouseButtonPress filter — narrow
        # to LeftButton to avoid disrupting napari's right/middle handlers.
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            try:
                self._annotator._on_subviewer_focused(self._orient)
            except Exception:
                pass
        return False  # don't consume — let napari/Qt handle as usual


class _CurationKeyFilter(QObject):
    """Event filter for X (atomic erase) and R (pick-refine) hotkeys.

    Installed on each subviewer's canvas.native widget so key events
    are caught at the Qt level before napari processes them.
    """

    def __init__(self, annotator, orient, parent=None):
        super().__init__(parent)
        self._annotator = annotator
        self._orient = orient

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        key = event.key()
        if key == Qt.Key_X:
            try:
                self._annotator._do_atomic_erase(self._orient)
            except Exception:
                pass
            return True  # consume the event
        if key == Qt.Key_R:
            try:
                self._annotator._do_pick_refine(self._orient)
            except Exception:
                pass
            return True  # consume the event
        return False


def _diag(tag: str, msg: str, *, stack: bool = False) -> None:
    """Diagnostic print with timestamp for lifecycle debugging."""
    t = _time.strftime("%H:%M:%S", _time.localtime())
    ms = int((_time.time() % 1) * 1000)
    print(f"[2D-DIAG {t}.{ms:03d}] {tag}: {msg}")
    if stack:
        _tb.print_stack(limit=6)


def _count_viewer_handlers(viewer) -> str:
    """Count handlers on key viewer event emitters."""
    parts = []
    for name in ("inserted", "removed"):
        try:
            emitter = getattr(viewer.layers.events, name)
            # napari EventEmitter stores callbacks — try common attributes
            n = "?"
            for attr in ("_slots", "callbacks", "_callbacks"):
                val = getattr(emitter, attr, None)
                if val is not None:
                    n = len(val)
                    break
            parts.append(f"{name}={n}")
        except Exception:
            parts.append(f"{name}=ERR")
    return " ".join(parts)


def _canvas_status(widget) -> str:
    """Check if subviewer canvas backends are alive or dead."""
    parts = []
    try:
        import sip
    except ImportError:
        return "(sip unavailable)"
    for orient in ("XY", "XZ", "YZ"):
        qtv = getattr(widget, f"qt_viewer_{orient.lower()}", None)
        if qtv is None:
            parts.append(f"{orient}=None")
        elif sip.isdeleted(qtv):
            parts.append(f"{orient}=DEAD")
        else:
            parts.append(f"{orient}=alive")
    return " ".join(parts)


class Curation2DAnnotator(NativeMultiviewAnnotator):
    """2D HITL curation annotator with lifecycle management.

    Subclasses NativeMultiviewAnnotator the same way Curation3DAnnotator
    does — overriding lifecycle methods while inheriting the base UI
    (3-panel multiview, combo boxes, Quick Annotate controls).
    """

    def __init__(self, viewer):
        _diag("__init__", f"BEFORE super().__init__ | viewer handlers: {_count_viewer_handlers(viewer)}")

        # --- Lifecycle guards (must exist before super().__init__) ---
        self._is_closing = False
        self._close_cleanup_done = False
        self._subviewers_closed = False
        self._session_token = 1

        # Slider callback storage — populated in _wire_signals override.
        self._slider_cbs = {}

        # --- Call parent __init__ ---
        # This creates managers, builds UI (including Quick Annotate
        # controls), wires signals, and refreshes layer options.
        super().__init__(viewer)

        # --- Deferred auto-load timers ---
        # Same pattern as Curation3DAnnotator: use a cancellable timer
        # so set_context() can cancel the init timer, preventing a
        # double load (SLOW PATH + redundant FAST PATH).
        from qtpy.QtCore import QTimer

        self._init_auto_load_timer = QTimer()
        self._init_auto_load_timer.setSingleShot(True)
        self._init_auto_load_timer.setInterval(100)
        self._init_auto_load_timer.timeout.connect(self._auto_load_if_ready)
        self._init_auto_load_timer.start()

        self._deferred_auto_load_timer = QTimer()
        self._deferred_auto_load_timer.setSingleShot(True)
        self._deferred_auto_load_timer.timeout.connect(self._auto_load_if_ready)

        # --- Drawing callback: detect edits on predicted slices ---
        self._on_slice_edited_callback = None  # Set by controller
        # --- Edit-state soft-lock (2026-05-27): controller injects a
        # callback (view, idx) -> status string so the per-orient buttons
        # can show emerald (accepted) / ice (finalized) + disable on
        # locked slices.  None until the controller wires it at launch. ---
        self._slice_status_callback = None
        if hasattr(self, "drawing_handler") and self.drawing_handler is not None:
            self.drawing_handler.post_sync_callback = self._on_draw_complete

        # --- Mode status indicator (in Quick Annotate panel) ---
        self._mode_status_label = None
        self._setup_mode_overlays()

        # --- 2D-specific hotkeys via Qt event filter (X=erase, R=refine) ---
        self._key_filters: list[tuple] = []  # (widget, filter) for teardown
        self._setup_2d_hotkeys()

        # --- Mode change listener: update overlays when mode changes ---
        self._mode_event_connections: list[tuple] = []
        # (connected after layers are loaded, in _load_selected_layers)

        # --- Layer insertion listener ---
        # Refresh layer options when new layers appear in the viewer.
        self._layer_inserted_cb = lambda _e: self._on_layer_inserted(_e)
        try:
            self.viewer.layers.events.inserted.connect(self._layer_inserted_cb)
        except Exception:
            pass

        # --- Per-subviewer focus-driven layer activation (2026-05-26) ---
        # Click/interaction on a subviewer canvas → that VM's selection.active
        # gets routed to the class-matching labels layer for that orientation,
        # mode = pan_zoom (user opts into paint via '2').  Sidesteps the
        # broken sync_active_layer cascade that lands cross-class layers on
        # wrong VMs.  See _setup_focus_filters, _on_subviewer_focused below.
        self._focused_orient: str | None = None
        self._focus_filters: list = []
        self._setup_focus_filters()

        _diag("__init__", f"AFTER all setup | viewer handlers: {_count_viewer_handlers(viewer)}")

    # ------------------------------------------------------------------
    # Signal wiring override — store slider refs for clean disconnect
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        """Extend parent to store slider callback references.

        The base class connects anonymous lambdas to slider.valueChanged
        that can't be disconnected. We intercept the sliders BEFORE
        super() connects them, store our own callbacks, and skip the
        base class's anonymous ones.
        """
        # Disconnect any existing slider connections first (defensive).
        for orient in ("XY", "XZ", "YZ"):
            slider = getattr(self, f"{orient.lower()}_slice_slider", None)
            if slider is not None:
                try:
                    slider.valueChanged.disconnect()
                except (TypeError, RuntimeError):
                    pass

        # Let parent wire everything (including anonymous slider lambdas).
        super()._wire_signals()

        # Now disconnect the anonymous lambdas and reconnect with
        # stored references that we can disconnect on cleanup.
        for orient in ("XY", "XZ", "YZ"):
            slider = getattr(self, f"{orient.lower()}_slice_slider", None)
            if slider is not None:
                try:
                    slider.valueChanged.disconnect()
                except (TypeError, RuntimeError):
                    pass
                cb = lambda value, o=orient: self._on_slice_slider_changed(o, value)
                self._slider_cbs[orient] = (slider, cb)
                slider.valueChanged.connect(cb)

    # ------------------------------------------------------------------
    # Session token gating (copied from Curation3DAnnotator)
    # ------------------------------------------------------------------

    def _invalidate_session_callbacks(self) -> None:
        """Invalidate deferred callbacks captured by old session token."""
        try:
            self._session_token = int(getattr(self, "_session_token", 0)) + 1
        except Exception:
            self._session_token = 1

    def _run_in_active_session(self, token: int, callback) -> None:
        """Run callback only when this widget is still in the active session."""
        if token != getattr(self, "_session_token", None):
            return
        if getattr(self, "_is_closing", False):
            return
        try:
            callback()
        except Exception:
            pass

    def _single_shot_in_session(self, delay_ms: int, callback) -> None:
        """Schedule callback guarded by current session token."""
        from qtpy.QtCore import QTimer

        token = int(getattr(self, "_session_token", 0))
        QTimer.singleShot(
            max(0, int(delay_ms)),
            lambda t=token, cb=callback: self._run_in_active_session(t, cb),
        )

    # ------------------------------------------------------------------
    # Deferred auto-load (same pattern as Curation3DAnnotator)
    # ------------------------------------------------------------------

    def _auto_load_if_ready(self) -> None:
        """Deferred auto-load: refresh combos and load layers into subviewers."""
        if getattr(self, "_is_closing", False):
            return
        try:
            if hasattr(self, "isVisible") and not self.isVisible():
                return
        except Exception:
            return
        self._refresh_layer_options()
        img = self.image_combo.currentText()
        lbl = self.labels_combo.currentText()
        if img and lbl:
            try:
                self._load_selected_layers()
            except Exception as exc:
                self.status_label.setText(f"Auto-load failed: {exc}")
        elif not img:
            self.status_label.setText(
                "No image layers found — add images to the viewer first."
            )
        elif not lbl:
            self.status_label.setText(
                "No labels layer found — add a labels layer first."
            )

    def _on_draw_complete(self) -> None:
        """Called after each drawing stroke on a subviewer.

        Notifies the controller that a slice has been edited, so it
        can transition predicted → curating status.
        """
        if getattr(self, "_is_closing", False):
            return
        cb = getattr(self, "_on_slice_edited_callback", None)
        _diag("_on_draw_complete", f"FIRED | callback={'set' if cb else 'None'}")
        if cb is not None:
            try:
                # Read the ACTUAL orientation from the DrawingHandler
                # (set during _on_mouse_drag), not the UI-selected view.
                dh = getattr(self, "drawing_handler", None)
                if dh is not None:
                    view = str(getattr(dh, "_active_edit_view", None) or "").upper()
                if not view or view not in {"XY", "XZ", "YZ"}:
                    view = str(getattr(self, "_active_selected_view", "XY")).upper()
                _diag("_on_draw_complete", f"calling controller callback with view={view}")
                cb(view)
            except Exception as exc:
                _diag("_on_draw_complete", f"callback FAILED: {exc}")

    def _schedule_deferred_auto_load(self, delay_ms: int) -> None:
        """Schedule deferred auto-load with cancellation support."""
        timer = getattr(self, "_deferred_auto_load_timer", None)
        if timer is None:
            from qtpy.QtCore import QTimer

            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(self._auto_load_if_ready)
            self._deferred_auto_load_timer = timer
        try:
            timer.stop()
        except Exception:
            pass
        timer.setInterval(max(0, int(delay_ms)))
        timer.start()

    def _on_layer_inserted(self, _event) -> None:
        """Schedule auto-load when new layers appear in the viewer."""
        if getattr(self, "_is_closing", False):
            return
        # Don't auto-load if we already have layers loaded.
        if self._active_image_name and self._active_labels_name:
            return
        self._schedule_deferred_auto_load(200)

    # ------------------------------------------------------------------
    # set_context override — deferred auto-load instead of immediate
    # ------------------------------------------------------------------

    def set_context(
        self,
        *,
        image_layer_name: str = "",
        labels_layer_name: str = "",
        auto_load: bool = True,
        target_view: str = "",
        selected_view: str = "",
    ) -> None:
        """Override to defer auto-load until dock is attached.

        Same pattern as Curation3DAnnotator: set combo selections
        without triggering immediate _load_selected_layers(), then
        schedule deferred auto-load via timer.
        """
        if target_view:
            self._active_target_mode = (
                "lateral"
                if str(target_view).lower().startswith("lat")
                else "axial"
            )
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
        if auto_load:
            # Cancel the __init__ auto-load timer to prevent double load.
            init_timer = getattr(self, "_init_auto_load_timer", None)
            if init_timer is not None:
                init_timer.stop()
            self._schedule_deferred_auto_load(150)

    # ------------------------------------------------------------------
    # _load_selected_layers override — fix slow path proxy bug
    # ------------------------------------------------------------------

    def _load_selected_layers(self) -> None:
        """Override with pre-load cleanup + post-load sync fix.

        Pre-load: disconnect stale per-layer rename callbacks from the
        SubviewerMirrorManager to prevent handler accumulation across
        dock reuse / patch switch cycles.  Matches what Curation3DAnnotator
        does (reset isolation, colormap authority, suppress flags).

        Post-load: reconnect sync to the REAL napari layer (not the
        SimpleNamespace proxy from the base class slow path).
        """
        # --- Pre-load cleanup ---
        # Reset sync guard flags on the old labels layer so
        # connect_label_sync can reconnect to the new layer after
        # super(). Without this, the flag persists on the layer object
        # and connect_label_sync returns early (no-op) even for a
        # different layer if Python reuses the same id().
        old_labels = self._active_labels_name
        if old_labels:
            try:
                old_layer = self._layer_by_name(self.viewer.layers, old_labels)
                if old_layer is not None:
                    for flag in ("_epicycle_sync_connected", "_epicycle_drag_connected"):
                        if hasattr(old_layer, flag):
                            setattr(old_layer, flag, False)
            except Exception:
                pass

        labels_name = self.labels_combo.currentText().strip()

        # --- Call parent's _load_selected_layers ---
        super()._load_selected_layers()

        # --- Post-load: reconnect sync to REAL layer (not proxy) ---
        if labels_name:
            lobj = self._layer_by_name(self.viewer.layers, labels_name)
            if isinstance(lobj, Labels):
                self.sync_manager.connect_label_sync(lobj)

        # --- 2D-specific constraints ---
        # Lock brush size to 1 for precision editing.
        self._enforce_brush_size_1()
        # Enclosed fill stays enabled — draw a circle, auto-fills interior,
        # then auto-increments to max(volume)+1 for the next label.
        # Connect mode change events for overlay indicators.
        self._connect_mode_events()
        self._update_mode_overlays()

        # C2 (review follow-up): clear stale _focused_orient on patch
        # switch.  Without this, the first click in the new patch is
        # treated as "same-orient re-click" and preserves whatever mode
        # the old layer was in.  Reset to None so the first click is a
        # clean first-focus (no mode reset, no glow until user opts in).
        self._focused_orient = None
        if hasattr(self, "_update_per_view_indicator"):
            try:
                self._update_per_view_indicator()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Mode indicator overlays
    # ------------------------------------------------------------------

    _MODE_DISPLAY = {
        "pan_zoom": ("PAN", "#888888"),
        "paint": ("PAINT", "#6abf69"),
        "erase": ("ERASE", "#ef5350"),
        "fill": ("FILL", "#d4a843"),
        "pick": ("PICK", "#80deea"),
    }

    def _setup_mode_overlays(self):
        """Create a mode status label in the Quick Annotate panel."""
        self._mode_status_label = None  # will be set up after draw_group exists

    def _ensure_mode_status_label(self):
        """Lazily insert a mode status label into the Quick Annotate panel."""
        if getattr(self, "_mode_status_label", None) is not None:
            return
        draw_group = getattr(self, "draw_group", None)
        if draw_group is None:
            return
        layout = draw_group.layout()
        if layout is None:
            return
        label = QLabel("Mode: —")
        label.setStyleSheet(
            "color: #888; font-size: 11px; font-weight: bold; "
            "padding: 2px 0px;"
        )
        # Insert at the top of the draw group (after title).
        layout.insertWidget(0, label)
        self._mode_status_label = label

    def _update_mode_overlays(self, *_args):
        """Refresh mode status label from current editable label layer mode."""
        if getattr(self, "_is_closing", False):
            return
        self._ensure_mode_status_label()
        label = getattr(self, "_mode_status_label", None)
        if label is None:
            return
        # Find the first editable label layer's mode.
        mode = "pan_zoom"
        editable = False
        for layer in self._all_subviewer_label_layers():
            if layer.editable:
                mode = str(getattr(layer, "mode", "pan_zoom"))
                editable = True
                break
        if not editable:
            # Hard-lock (2026-05-28): if the active-bucket slice is locked
            # because it's FINALIZED (accepted/staged/committed), say so +
            # point the user at Revert — instead of a bare "LOCKED" that
            # could be mistaken for an inactive bucket.
            if self._active_bucket_slice_is_finalized():
                text = "Mode: LOCKED — Revert to edit"
                color = "#80deea"  # ice — matches the finalized button tint
            else:
                text = "Mode: LOCKED"
                color = "#555555"
        else:
            display_text, color = self._MODE_DISPLAY.get(mode, (mode.upper(), "#888888"))
            text = f"Mode: {display_text}"
        label.setText(text)
        label.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: bold; "
            "padding: 2px 0px;"
        )
        # Refresh per-view button glow whenever any labels layer's mode
        # changes (mode events on subviewer layers are wired in
        # _connect_mode_events).  Catches the user-presses-'2' transition
        # and lights up the focused view's buttons in real time.
        if hasattr(self, "_update_per_view_indicator"):
            try:
                self._update_per_view_indicator()
            except Exception:
                pass

    def _active_bucket_slice_is_finalized(self) -> bool:
        """Hard-lock helper (2026-05-28): True if the currently-displayed
        slice of any ACTIVE-bucket orient is finalized (accepted/staged/
        committed).  Used to give the LOCKED mode overlay a finalized-
        specific message.  Training-lock takes precedence (handled by its
        own overlay), so this only matters when not training-locked."""
        if getattr(self, "_training_lock_active", False):
            return False
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        if target == "lateral":
            orients = ("XY",)
        elif target == "axial":
            orients = ("XZ", "YZ")
        else:
            return False
        for o in orients:
            try:
                if self._edit_state_for_orient(o) != "editable":
                    return True
            except Exception:
                continue
        return False

    def _connect_mode_events(self):
        """Connect label layer mode events to update status label."""
        self._disconnect_mode_events()
        for layer in self._all_subviewer_label_layers():
            try:
                cb = lambda _e, _s=self: _s._update_mode_overlays()
                layer.events.mode.connect(cb)
                self._mode_event_connections.append((layer, cb))
            except Exception:
                pass

    def _disconnect_mode_events(self):
        """Disconnect mode event listeners."""
        for layer, cb in getattr(self, "_mode_event_connections", []):
            try:
                layer.events.mode.disconnect(cb)
            except Exception:
                pass
        self._mode_event_connections = []

    # ------------------------------------------------------------------
    # 2D-specific hotkeys: E = atomic instance erase, R = pick-and-refine
    # ------------------------------------------------------------------

    def _setup_2d_hotkeys(self):
        """Install Qt event filters on subviewer canvases for X/R hotkeys.

        C2 (2026-05-22): defensive against handler accumulation across
        close→relaunch cycles.  The previous design tracked installed
        filters only in ``self._key_filters``; if a teardown silently
        failed to remove a filter (the ``except Exception: pass`` at
        the bottom of :meth:`_teardown_2d_hotkeys` will swallow any
        Qt-side error), the tracking list was wiped while the filter
        stayed installed on the canvas — the next setup then added a
        SECOND filter to the same canvas, causing X (atomic erase)
        and R (pick-refine) to fire 2-3× per keystroke.

        Fix: each canvas now ALSO stores its installed filter via an
        attribute ``_epicycle_curation_keyfilter``.  Setup checks this
        attribute and explicitly removes any prior filter before
        installing a new one — independent of whether the tracking
        list survived.  The canvas is the source of truth.
        """
        self._teardown_2d_hotkeys()
        for orient, qt_viewer in [
            ("XY", self.qt_viewer_xy),
            ("XZ", self.qt_viewer_xz),
            ("YZ", self.qt_viewer_yz),
        ]:
            try:
                canvas = qt_viewer.canvas.native
                # C2: defensive canvas-side cleanup of any prior filter.
                prior = getattr(canvas, "_epicycle_curation_keyfilter", None)
                if prior is not None:
                    try:
                        canvas.removeEventFilter(prior)
                    except Exception:
                        pass
                    try:
                        delattr(canvas, "_epicycle_curation_keyfilter")
                    except Exception:
                        pass
                filt = _CurationKeyFilter(self, orient, parent=canvas)
                canvas.installEventFilter(filt)
                try:
                    canvas._epicycle_curation_keyfilter = filt
                except Exception:
                    pass
                self._key_filters.append((canvas, filt))
                _diag("_setup_2d_hotkeys", f"installed key filter on {orient}")
            except Exception as exc:
                _diag("_setup_2d_hotkeys", f"FAILED on {orient}: {exc}")

    def _teardown_2d_hotkeys(self):
        """Remove Qt event filters for 2D hotkeys.

        C2 (2026-05-22): also clears the per-canvas
        ``_epicycle_curation_keyfilter`` attribute so a subsequent
        setup doesn't see a stale reference.  See :meth:`_setup_2d_hotkeys`.
        """
        for canvas, filt in getattr(self, "_key_filters", []):
            try:
                canvas.removeEventFilter(filt)
            except Exception:
                pass
            try:
                if getattr(canvas, "_epicycle_curation_keyfilter", None) is filt:
                    delattr(canvas, "_epicycle_curation_keyfilter")
            except Exception:
                pass
        self._key_filters = []

    def _vm_for_orient(self, orient):
        """Return the ViewerModel for a given orientation."""
        return {"XY": self.viewer_model_xy, "XZ": self.viewer_model_xz, "YZ": self.viewer_model_yz}.get(orient)

    def _do_atomic_erase(self, orient):
        """X key handler: erase all pixels of the hovered label on this slice."""
        if getattr(self, "_is_closing", False):
            return
        vm = self._vm_for_orient(orient)
        if vm is None:
            return
        layer = self._find_editable_label_layer(vm, orient)
        if layer is None:
            return

        label_val = self._get_label_at_cursor(layer, vm)
        _diag("_do_atomic_erase", f"orient={orient} label_val={label_val}")
        if label_val is None or label_val == 0:
            return

        try:
            data = np.asarray(layer.data)
            s_idx = int(vm.dims.current_step[0])
            if s_idx < 0 or s_idx >= data.shape[0]:
                return
            data_2d = data[s_idx]
            mask = data_2d == label_val
            if not np.any(mask):
                return
            data_2d[mask] = 0
            layer.refresh()
            if hasattr(self, "sync_manager"):
                self.sync_manager.sync_to_main_view(layer, orient, vm)
            _diag("_do_atomic_erase", f"erased label {label_val} on {orient} slice {s_idx}")
        except Exception as exc:
            _diag("_do_atomic_erase", f"failed: {exc}")

    def _do_pick_refine(self, orient):
        """R key handler: pick label under cursor, switch to paint mode."""
        if getattr(self, "_is_closing", False):
            return
        vm = self._vm_for_orient(orient)
        if vm is None:
            return
        layer = self._find_editable_label_layer(vm, orient)
        if layer is None:
            return

        label_val = self._get_label_at_cursor(layer, vm)
        _diag("_do_pick_refine", f"orient={orient} label_val={label_val}")
        if label_val is None or label_val == 0:
            return

        # Set selected_label everywhere.
        try:
            layer.selected_label = label_val
            for lname in (getattr(self, "_active_labels_names", None) or []):
                main_layer = self._layer_by_name(self.viewer.layers, lname)
                if isinstance(main_layer, Labels):
                    main_layer.selected_label = label_val
        except Exception as exc:
            _diag("_do_pick_refine", f"set label failed: {exc}")
            return

        # Force paint mode with immediate + deferred pass.
        for sub_layer in self._all_subviewer_label_layers():
            if sub_layer.editable:
                try:
                    sub_layer.mode = "paint"
                except Exception:
                    pass
        self._update_mode_overlays()

        from qtpy.QtCore import QTimer
        def _ensure_paint():
            if getattr(self, "_is_closing", False):
                return
            for sub_layer in self._all_subviewer_label_layers():
                if sub_layer.editable and sub_layer.mode != "paint":
                    sub_layer.mode = "paint"
            self._update_mode_overlays()
        QTimer.singleShot(100, _ensure_paint)

    def _get_label_at_cursor(self, layer, viewer_model):
        """Read the label value at the current cursor position.

        Reads directly from the layer's data array using the subviewer's
        dims and cursor position. Avoids napari's get_value() which
        doesn't handle transposed array views correctly for drawn labels.
        """
        try:
            cursor_pos = viewer_model.cursor.position
            if cursor_pos is None or len(cursor_pos) < 3:
                return None
            data = np.asarray(layer.data)
            if data.ndim != 3:
                return None
            # Subviewers always slice along axis 0.
            s_idx = int(viewer_model.dims.current_step[0])
            if s_idx < 0 or s_idx >= data.shape[0]:
                return None
            # cursor_pos[1] = row (y), cursor_pos[2] = col (x) in the 2D view.
            r = int(round(cursor_pos[1]))
            c = int(round(cursor_pos[2]))
            r = max(0, min(r, data.shape[1] - 1))
            c = max(0, min(c, data.shape[2] - 1))
            val = int(data[s_idx, r, c])
            # If val is 0, check if the slice has ANY labels at all.
            if val == 0:
                slice_data = data[s_idx]
                slice_max = int(np.max(slice_data))
                unique_count = len(np.unique(slice_data))
                _diag("_get_label_at_cursor",
                      f"val=0 | s_idx={s_idx} r={r} c={c} | "
                      f"layer={layer.name} | shape={data.shape} | "
                      f"slice_max={slice_max} unique_labels={unique_count} | "
                      f"data_id={id(data)} layer_data_id={id(layer.data)}")
            else:
                _diag("_get_label_at_cursor",
                      f"val={val} | s_idx={s_idx} r={r} c={c} | "
                      f"layer={layer.name} | shape={data.shape}")
            return val
        except Exception as exc:
            _diag("_get_label_at_cursor", f"failed: {exc}")
            return None

    def _find_editable_label_layer(self, viewer_model, orient):
        """Find the correct label layer for this orientation in a subviewer.

        In axial mode, both XZ and YZ volumes exist in each subviewer
        (transposed). We must find the layer whose BASE name matches
        the requested orientation — not just any editable layer.

        Matching rules:
        - orient=XY: base must contain '_lateral_' or '_work_labels_3d'
        - orient=XZ: base must contain '_axial_xz_'
        - orient=YZ: base must contain '_axial_yz_'
        """
        orient_upper = orient.upper()
        suffix = f"_{orient_upper}"

        # Map orient to the expected base name fragment.
        _BASE_TAGS = {
            "XY": ("_lateral_", "_work_labels_3d"),
            "XZ": ("_axial_xz_",),
            "YZ": ("_axial_yz_",),
        }
        tags = _BASE_TAGS.get(orient_upper, ())

        # Preferred: match by known active label names, BUT only if the
        # base name is compatible with the requested orient (e.g., don't
        # return axial_xz_3d_YZ when looking for YZ — that's XZ data).
        for lname in (getattr(self, "_active_labels_names", None) or []):
            # Check if base name matches orient (or is lateral/legacy).
            lname_lower = lname.lower()
            name_ok = any(tag in lname_lower for tag in tags)
            if not name_ok:
                continue
            layer = self._layer_by_name(viewer_model.layers, f"{lname}_{orient_upper}")
            if isinstance(layer, Labels):
                return layer

        # Fallback: find layer whose name ends with _{orient} AND
        # whose base name contains the correct view tag.
        # Diagnostic: list ALL Labels layers in this subviewer.
        all_labels_in_vm = [(l.name, l.editable) for l in viewer_model.layers if isinstance(l, Labels)]
        _diag("_find_editable_label_layer",
              f"orient={orient_upper} | tags={tags} | suffix={suffix} | "
              f"all_labels={all_labels_in_vm}")
        for layer in viewer_model.layers:
            if not isinstance(layer, Labels) or not layer.name.endswith(suffix):
                continue
            name_lower = layer.name.lower()
            matched_tag = any(tag in name_lower for tag in tags)
            _diag("_find_editable_label_layer",
                  f"  candidate: {layer.name} | matched_tag={matched_tag}")
            if matched_tag:
                return layer

        _diag("_find_editable_label_layer", "NO MATCH — returning None")
        return None

    # ------------------------------------------------------------------
    # Axial decoupling: separate Segment/Accept buttons for XZ and YZ
    # ------------------------------------------------------------------

    def _build_draw_group(self):
        """Override: rearrange Segment/Accept buttons into 3 rows × 2 columns,
        add section separators, and restyle dropdowns + neutral buttons.

        2026-05-27 redesign:
        - Each per-orientation row has Segment + Accept side by side, tinted
          left edge in the orient color (yellow/cyan/magenta from crosshair).
        - All 6 buttons always visible.  Bucket mode toggles enabled state.
        - Active paint surface: button background fills with orient color at
          ~25% opacity (no separate amber accent — orient color IS the signal).
        - Section labels + thin horizontal separators between Annotate / Overlay /
          Tools groups.  Drops the heavy default-napari gray on dropdowns +
          neutral buttons.
        """
        from qtpy.QtWidgets import QWidget

        group = super()._build_draw_group()
        layout = group.layout()

        # Tighten group-box padding so the new section structure breathes
        # without bloating the panel.
        try:
            layout.setContentsMargins(8, 10, 8, 10)
            layout.setSpacing(6)
        except Exception:
            pass

        # 2026-06-01: per-section `?` help button on the "Quick Annotate"
        # group, appended inline with the crosshair_check row via the
        # base class's QHBoxLayout (right-aligned via addStretch).  Uses
        # the shared help_button() helper from ui_shell.theme.
        #
        # NOTE: "Refinement Navigation" group help is INTENTIONALLY NOT
        # wired here.  Pre-commit audit 2026-06-01 found that the
        # _focused_mode_group is gated on `_3d_curation_mode=True` which
        # the 2D HITL annotator never sets — so the group is permanently
        # hidden in this workflow.  A help button there would be
        # unreachable.  The base class still exposes
        # `_focused_toggle_row_layout` as forward-compat scaffolding for
        # the future Step 3 / 3D annotator help wiring.
        try:
            self._crosshair_row_layout.addWidget(_help_button(
                self,
                guide_path=str(
                    _ANNOTATOR_GUIDES_DIR / "annotator_2d_quick_annotate.md"
                ),
                title="Quick Annotate — 2D HITL Annotator",
                tooltip="Help: Quick Annotate",
            ))
        except Exception:
            # Defensive: base class didn't expose the layout (older build).
            pass

        # Create axial-specific buttons (still per-orientation).
        self.segment_xz_btn = QPushButton("Segment XZ")
        self.segment_yz_btn = QPushButton("Segment YZ")
        self.accept_xz_btn = QPushButton("Accept XZ")
        self.accept_yz_btn = QPushButton("Accept YZ")

        # Pull the base class's inherited XY buttons out of the layout so
        # we can re-pair them into _xy_row.  Base added them as separate
        # full-width widgets at indices 2 and 3 (after crosshair + patch row).
        layout.removeWidget(self.segment_mode_btn)
        layout.removeWidget(self.accept_slice_btn)

        # Build 3 per-orientation rows: each has Segment + Accept side by side.
        def _make_row(seg_btn: QPushButton, acc_btn: QPushButton) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            h.addWidget(seg_btn)
            h.addWidget(acc_btn)
            return w

        self._xy_row = _make_row(self.segment_mode_btn, self.accept_slice_btn)
        self._xz_row = _make_row(self.segment_xz_btn, self.accept_xz_btn)
        self._yz_row = _make_row(self.segment_yz_btn, self.accept_yz_btn)

        # Section header + separator helpers.
        def _make_section_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(_SECTION_LABEL_STYLE)
            return lbl

        def _make_separator() -> QFrame:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setFrameShadow(QFrame.Plain)
            sep.setStyleSheet(_SEP_LINE_STYLE)
            sep.setFixedHeight(1)
            return sep

        # Insert per-orient rows + section structure.  Base layout order at
        # entry: [0]=crosshair_check, [1]=patch_row, then overlay/tools below.
        # We insert between patch_row and the overlay block:
        #   [2] separator
        #   [3] "Annotate" label
        #   [4] XY row
        #   [5] XZ row
        #   [6] YZ row
        #   [7] separator   (overlay block starts after this)
        self._sep_top = _make_separator()
        self._annotate_label = _make_section_label("Annotate")
        self._sep_after_rows = _make_separator()

        layout.insertWidget(2, self._sep_top)
        layout.insertWidget(3, self._annotate_label)
        layout.insertWidget(4, self._xy_row)
        layout.insertWidget(5, self._xz_row)
        layout.insertWidget(6, self._yz_row)
        layout.insertWidget(7, self._sep_after_rows)

        # Apply initial per-orient tint stylesheets (faint/at-rest state).
        self.segment_mode_btn.setText("Segment XY")
        self.accept_slice_btn.setText("Accept XY")
        for btn in (self.segment_mode_btn, self.accept_slice_btn):
            btn.setStyleSheet(_orient_button_style("XY", paint_active=False))
        for btn in (self.segment_xz_btn, self.accept_xz_btn):
            btn.setStyleSheet(_orient_button_style("XZ", paint_active=False))
        for btn in (self.segment_yz_btn, self.accept_yz_btn):
            btn.setStyleSheet(_orient_button_style("YZ", paint_active=False))

        # Restyle non-orient buttons (Switch, Advanced Features..., Multiviewer
        # Settings...) and dropdowns (Patch combo, Overlay combo) to drop the
        # heavy default-napari gray look.
        for attr in (
            "switch_patch_btn",
            "open_advanced_features_btn",
            "_open_multiviewer_settings_btn",
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                try:
                    btn.setStyleSheet(_NEUTRAL_BTN_STYLE)
                except Exception:
                    pass
        for attr in ("patch_context_combo", "overlay_mode_combo"):
            combo = getattr(self, attr, None)
            if combo is not None:
                try:
                    combo.setStyleSheet(_DROPDOWN_STYLE)
                except Exception:
                    pass

        # release hide list 2026-10-12: the Advanced Features dock is not
        # built for the 2D annotator; hidden (reversible), handler kept.
        self.open_advanced_features_btn.setVisible(False)

        return group

    def _wire_signals(self):
        """Override: wire axial buttons after base class wiring."""
        super()._wire_signals()
        self.segment_xz_btn.clicked.connect(lambda: self._request_segment_orientation("XZ"))
        self.segment_yz_btn.clicked.connect(lambda: self._request_segment_orientation("YZ"))
        self.accept_xz_btn.clicked.connect(lambda: self._request_accept_orientation("XZ"))
        self.accept_yz_btn.clicked.connect(lambda: self._request_accept_orientation("YZ"))

    def _request_segment_orientation(self, orient):
        """Segment a single orientation (XZ or YZ)."""
        self._request_segment_slice(orient)

    def _request_accept_orientation(self, orient):
        """Accept a single orientation (XZ or YZ)."""
        orient = str(orient).upper()
        cb = getattr(self, "_accept_slice_callback", None)
        if not callable(cb):
            self.status_label.setText("Accept callback not available.")
            return
        vm = self._vm_for_orient(orient)
        if vm is None:
            self.status_label.setText(f"No viewer for {orient}.")
            return
        s_idx = self._safe_first_dim_index(vm)
        try:
            result = cb(orient, slice_index=s_idx) or {}
        except Exception as exc:
            self.status_label.setText(f"Accept failed for {orient}: {exc}")
            return
        ok = bool(result.get("ok", False))
        msg = str(result.get("message", "")).strip()
        if msg:
            self.status_label.setText(msg)
        else:
            self.status_label.setText(f"{'Accepted' if ok else 'No mask for'} {orient} slice {s_idx}.")

    def _refresh_segment_button_text(self):
        """Override: enable/disable per-view buttons based on bucket mode.

        2026-05-26 redesign: all 6 buttons (Segment + Accept × XY/XZ/YZ) are
        always visible.  The inactive bucket's buttons are disabled (Qt's
        native dimmed state) rather than hidden.  Stable panel size + better
        discoverability + per-orient tints stay visible on disabled buttons.
        """
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        # All rows always visible; enable/disable + edit-state styling is
        # delegated to _update_per_view_indicator (single source of truth
        # that factors BOTH bucket AND per-slice edit-state).  2026-05-27:
        # the inline bucket-only enable block was removed so bucket and
        # edit-state can't fight over the enabled flag.
        for row_attr in ("_xy_row", "_xz_row", "_yz_row"):
            row = getattr(self, row_attr, None)
            if row is not None:
                row.setVisible(True)
        # If focused orient is now in the disabled bucket, clear focus so the
        # glow indicator doesn't get stuck on an unreachable button.
        focused = getattr(self, "_focused_orient", None)
        if focused == "XY" and target == "axial":
            self._focused_orient = None
        elif focused in {"XZ", "YZ"} and target == "lateral":
            self._focused_orient = None
        # Refresh per-view indicator (sets enable + style for all 6 buttons).
        if hasattr(self, "_update_per_view_indicator"):
            try:
                self._update_per_view_indicator()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Training lock (Step 2 only — 2026-05-27)
    # ------------------------------------------------------------------

    def _apply_edit_constraints(self) -> None:
        """Override the base's per-orient editability derivation.

        Audit fix (2026-05-27, second-pass review #8): the base's
        ``_apply_edit_constraints`` at ``native_multiview_widget.py:2095``
        sets ``layer.editable = bool(allow and visible)`` based on focused
        orient + bucket — it has NO awareness of the training lock.  If
        ANY code path triggers ``_apply_edit_constraints`` while the
        lock is active (napari layer events, focus change, target-mode
        flip), the lock's editable=False on those layers is silently
        overridden.

        Step-2-only fix: override here in ``Curation2DAnnotator`` to
        early-return when the training lock is active.  The base method
        is unchanged; ``Curation3DAnnotator`` (Step 3, inherits from
        the base directly) is unaffected.

        When the lock clears, ``set_training_lock(False)`` restores
        each layer's snapshotted editable state and ALSO calls
        ``_refresh_segment_button_text`` so per-bucket button logic
        re-derives — and any subsequent ``_apply_edit_constraints``
        call (via mode change, focus change, etc.) will run normally
        because ``_training_lock_active`` is False.
        """
        if getattr(self, "_training_lock_active", False):
            print("[STEP2-LOCK] _apply_edit_constraints BLOCKED — "
                  "training lock is active (would override lock state)")
            return
        super()._apply_edit_constraints()
        # Hard-lock (2026-05-28): after the base sets per-bucket
        # editability, downgrade any subviewer layer whose CURRENTLY-
        # displayed slice is finalized (accepted/staged/committed) to
        # editable=False.  Finalized slices can't be painted until the
        # user Reverts in the panel.  This consolidation point ensures
        # the lock survives every _apply_edit_constraints trigger (mode
        # change, focus change, patch load) — no hydra.
        self._enforce_finalized_slice_lock()
        # Main-canvas read-only (2026-05-28): the main napari canvas is
        # visualization-only in Step 2 — re-lock it here too so no
        # editability-setter leaves it paintable.
        self._lock_main_canvas_readonly()

    def _set_local_work_label_visibility(self, patch_prefix: str) -> None:
        """Override (2026-05-28): the base sets the MAIN-viewer work_labels
        ``editable = bool(visible)`` during layer load.  In Step 2 the main
        canvas is visualization-only, so re-lock it read-only after the
        base runs.  Step-2-only — does NOT touch the base method (Step 3's
        Curation3DAnnotator inherits the base unchanged, keeping its main
        viewer editable for 3D curation drawing)."""
        super()._set_local_work_label_visibility(patch_prefix)
        self._lock_main_canvas_readonly()

    def _lock_main_canvas_readonly(self) -> None:
        """Force every MAIN-viewer work_labels layer non-editable
        (editable=False + pan_zoom).  Step 2 draws on the subviewers; the
        main napari canvas is for visualization.  Iterates ``self.viewer.layers``
        (the main viewer) — the subviewer mirror layers live in separate
        ViewerModels and are NOT touched here (their editability is the
        finalized hard-lock's job).  Subviewer→main sync writes ``.data``
        directly, so read-only does not break it."""
        from napari.layers import Labels
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        try:
            layers = list(viewer.layers)
        except Exception:
            return
        for layer in layers:
            try:
                if not isinstance(layer, Labels):
                    continue
                if "_work_labels_" not in str(getattr(layer, "name", "")):
                    continue
                if getattr(layer, "editable", False):
                    layer.editable = False
                if getattr(layer, "mode", None) != "pan_zoom":
                    layer.mode = "pan_zoom"
            except Exception:
                continue

    def _enforce_finalized_slice_lock(self) -> None:
        """Hard-lock finalized slices (2026-05-28).

        Sets ``editable`` on each ACTIVE-bucket subviewer label layer based
        on whether its currently-displayed slice is finalized:
          - finalized (accepted/staged/committed)  -> editable=False + pan_zoom
          - editable (predicted/curating/none)     -> editable=True

        EDITABILITY-ONLY: does NOT touch ``visible`` (so it's safe to call
        on every scroll without resetting the user's a/s/d/f visibility
        toggles).  Only touches ACTIVE-bucket layers — inactive-bucket
        layers were already locked by the base ``_apply_edit_constraints``,
        so we leave them alone.  Bidirectional: re-enables an editable slice
        when scrolling off a finalized one.

        Reuses :meth:`_edit_state_for_orient` (the same per-slice status
        lookup the soft-lock visual uses).  Skipped during the training
        lock (``set_training_lock`` owns editability then).

        Threshold: accepted + staged + committed are all locked (i.e.
        anything _edit_state_for_orient does NOT classify as "editable").
        """
        if getattr(self, "_training_lock_active", False):
            return
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        if target == "lateral":
            active = {"XY"}
        elif target == "axial":
            active = {"XZ", "YZ"}
        else:  # 3d_curation — not the normal Step 2 path
            active = {"XY", "XZ", "YZ"}
        for layer in self._all_subviewer_label_layers():
            name = str(getattr(layer, "name", ""))
            try:
                _, orient = name.rsplit("_", 1)
                orient = orient.upper()
            except Exception:
                continue
            if orient not in active:
                continue  # inactive bucket — base already locked it
            try:
                if not getattr(layer, "visible", True):
                    continue  # hidden layer — don't force editability
                slice_editable = self._edit_state_for_orient(orient) == "editable"
                if bool(getattr(layer, "editable", False)) != slice_editable:
                    layer.editable = slice_editable
                    print(f"[HARD-LOCK] {orient} editable={slice_editable} "
                          f"(slice {'editable' if slice_editable else 'FINALIZED'})")
                if not slice_editable and getattr(layer, "mode", None) != "pan_zoom":
                    layer.mode = "pan_zoom"
            except Exception as exc:
                print(f"[HARD-LOCK] enforce failed for {name!r}: "
                      f"{type(exc).__name__}: {exc}")

    def set_training_lock(self, locked: bool) -> None:
        """Lock/unlock the annotator's drawing + segmenting surfaces while
        a training job is in flight.

        Called from ``TrainingAnnotationController._set_training_ui_locked``
        right before training launches and again at clean exit / cancel /
        error.  While locked:

          * EVERY Labels layer in the main napari viewer AND in all 3
            subviewers gets ``editable=False`` and ``mode="pan_zoom"``.
            We do NOT trust per-mode state because the annotator's
            existing logic re-derives editability based on focused orient
            and bucket — so the safest path is to forcibly lock all
            Labels layers and let the post-unlock refresh restore
            correctness.
          * All 6 per-orient Segment / Accept buttons are disabled.
          * The Mode overlay label is repainted to amber "LOCKED
            (TRAINING)" so the user has VISIBLE feedback about WHY
            painting + segmenting are blocked.

        Pan / zoom / scroll-through-slices remain unchanged so the user
        can still inspect what's happening across the patch while
        training runs.

        Idempotent: lock_training_lock(True) twice is safe; calling it
        with locked=False from an idle state is also safe (restore is
        defensive against missing attrs).

        Scope: this method lives on ``Curation2DAnnotator``, NOT on the
        base ``NativeMultiviewAnnotator``.  Step 3's
        ``Curation3DAnnotator`` does NOT inherit this and is unaffected.
        """
        print(f"[STEP2-LOCK] Curation2DAnnotator.set_training_lock locked={locked}")

        # Snapshot main viewer + subviewer Labels layers we touch, so we
        # only restore what we changed.  Stored on self so a second
        # lock(True) doesn't blow away the originals.
        if locked:
            if getattr(self, "_training_lock_active", False):
                # Already locked — idempotent no-op.  Don't re-snapshot,
                # don't re-touch.
                return
            self._training_lock_active = True
            self._training_lock_snapshot = []
            for layer in self._collect_lockable_label_layers():
                try:
                    prior_editable = bool(getattr(layer, "editable", False))
                    prior_mode = str(getattr(layer, "mode", "pan_zoom"))
                    self._training_lock_snapshot.append(
                        (layer, prior_editable, prior_mode)
                    )
                    layer.editable = False
                    if hasattr(layer, "mode"):
                        layer.mode = "pan_zoom"
                except Exception as exc:
                    print(f"[STEP2-LOCK] failed to lock layer "
                          f"{getattr(layer, 'name', '?')!r}: "
                          f"{type(exc).__name__}: {exc}")
            self._lock_per_orient_buttons(locked=True)
            self._set_mode_overlay_training_lock(True)
        else:
            if not getattr(self, "_training_lock_active", False):
                # Already unlocked — no-op.  Avoids accidentally
                # forcing editable=True on layers we never touched.
                return
            self._training_lock_active = False
            snapshot = getattr(self, "_training_lock_snapshot", None) or []
            for layer, prior_editable, prior_mode in snapshot:
                # Layer may have been removed during the lock period
                # (e.g. patch switch was attempted but blocked elsewhere).
                # Best-effort restore; ignore failures.
                try:
                    layer.editable = prior_editable
                    if prior_mode and hasattr(layer, "mode"):
                        layer.mode = prior_mode
                except Exception as exc:
                    print(f"[STEP2-LOCK] failed to unlock layer "
                          f"{getattr(layer, 'name', '?')!r}: "
                          f"{type(exc).__name__}: {exc}")
            self._training_lock_snapshot = []
            self._lock_per_orient_buttons(locked=False)
            self._set_mode_overlay_training_lock(False)
            # Refresh per-orient button enable state in case bucket
            # focus changed during the lock period — re-derive via the
            # existing helper (despite its name, this method ALSO
            # re-enables/disables the per-orient buttons based on the
            # current bucket — see method docstring).
            if hasattr(self, "_refresh_segment_button_text"):
                try:
                    self._refresh_segment_button_text()
                except Exception:
                    pass
            # Hard-lock (2026-05-28, audit defensive): after restoring the
            # pre-training editable snapshot, re-assert BOTH locks.  The
            # snapshot restore (which includes the main-viewer work_labels)
            # could wrongly re-enable a slice/canvas that should stay locked.
            # Today redundant (training only transitions slices INTO
            # finalized states, whose snapshot was already editable=False,
            # and the main canvas is always read-only), but re-enforcing
            # makes the locks robust rather than invariant-dependent.
            try:
                self._enforce_finalized_slice_lock()
            except Exception:
                pass
            try:
                self._lock_main_canvas_readonly()
            except Exception:
                pass

    def _collect_lockable_label_layers(self):
        """Return every Labels layer in main + 3 subviewers that we
        should force editable=False on during training.

        Reads-only from the base class (`_all_subviewer_label_layers`).
        Pulls main-viewer Labels via the main napari viewer's layer list.
        Order is irrelevant; we just need a deduplicated list."""
        from napari.layers import Labels
        out = []
        seen = set()

        # Main viewer Labels layers.
        main_viewer = getattr(self, "viewer", None)
        if main_viewer is not None:
            try:
                for layer in main_viewer.layers:
                    if isinstance(layer, Labels) and id(layer) not in seen:
                        out.append(layer)
                        seen.add(id(layer))
            except Exception as exc:
                print(f"[STEP2-LOCK] main-viewer scan failed: "
                      f"{type(exc).__name__}: {exc}")

        # Subviewer Labels layers (via base-class helper).
        try:
            for layer in self._all_subviewer_label_layers() or []:
                if isinstance(layer, Labels) and id(layer) not in seen:
                    out.append(layer)
                    seen.add(id(layer))
        except Exception as exc:
            print(f"[STEP2-LOCK] subviewer scan failed: "
                  f"{type(exc).__name__}: {exc}")

        return out

    def _lock_per_orient_buttons(self, *, locked: bool) -> None:
        """Disable/enable the 6 per-orient Segment + Accept buttons.

        On unlock, the existing ``_refresh_segment_button_state`` re-
        derives the right enable/disable state per bucket (axial vs
        lateral focus), so we don't need to remember prior state — just
        force-disable on lock and trigger a refresh on unlock.
        """
        names = (
            "segment_mode_btn",
            "accept_slice_btn",
            "segment_xz_btn",
            "accept_xz_btn",
            "segment_yz_btn",
            "accept_yz_btn",
        )
        for attr in names:
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            if hasattr(btn, "setEnabled"):
                try:
                    btn.setEnabled(not locked)
                except Exception as exc:
                    print(f"[STEP2-LOCK] failed to set {attr!r} "
                          f"enabled={not locked}: "
                          f"{type(exc).__name__}: {exc}")

    def _set_mode_overlay_training_lock(self, locked: bool) -> None:
        """Repaint the Mode status label to a distinct 'LOCKED (TRAINING)'
        state when locked=True; on locked=False let the existing
        ``_update_mode_overlays`` reclaim the label.

        We don't permanently override `_update_mode_overlays` — we set the
        label directly and set a `_training_lock_overlay_active` flag the
        base's update path can check (or ignore — the next mode event
        will re-paint, but during the lock there shouldn't be a mode
        event because all layers are pan_zoom and the user can't paint)."""
        self._training_lock_overlay_active = bool(locked)
        label = getattr(self, "_mode_status_label", None)
        if label is None:
            return
        if locked:
            try:
                label.setText("Mode: LOCKED (TRAINING)")
                label.setStyleSheet(
                    "color: #f0a040; font-size: 11px; font-weight: bold; "
                    "padding: 2px 0px;"
                )
            except Exception:
                pass
        else:
            # Let the next _update_mode_overlays call repaint with the
            # current actual mode.  Trigger one immediately.
            try:
                self._update_mode_overlays()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Channel keybinding override — toggle ALL work_labels in axial mode
    # ------------------------------------------------------------------

    def _setup_channel_keybindings(self) -> None:
        """Override: in axial mode, the labels toggle key hides/shows
        ALL work_labels layers (XZ + YZ), not just the one in
        _active_labels_names.

        Calls the base class implementation first, then replaces the
        toggle function for the labels key slot with one that covers
        all work_labels layers.
        """
        # Let base class set up a/s/d/f bindings normally.
        super()._setup_channel_keybindings()

        # Find which key index corresponds to labels.
        image_names = list(getattr(self, "_active_image_layer_names", []) or [])
        labels_idx = len(image_names)  # first labels key
        if labels_idx >= len(self._CHANNEL_KEYS):
            return  # no key slot for labels

        labels_key = self._CHANNEL_KEYS[labels_idx]

        # Build the list of ALL work_labels layer names in the main viewer.
        all_wl_names = []
        for lyr in self.viewer.layers:
            lyr_name = str(getattr(lyr, "name", ""))
            if "_work_labels_" in lyr_name and lyr_name not in all_wl_names:
                all_wl_names.append(lyr_name)

        if len(all_wl_names) <= 1:
            return  # single layer — base class toggle is fine

        def _toggle_all_labels(_viewer):
            """Toggle ALL work_labels layers + their subviewer copies.

            Directional save/restore.  Save mode BEFORE
            hiding (uncorrupted) and restore AFTER showing (overrides
            napari's corrupted _visible_mode restore on the mirrors).
            Reentrancy guard: drop repeat keypresses that
            arrive while the previous toggle's cascade is still
            settling.  See base class ``_setup_channel_keybindings``
            for the canonical rationale and commits a922e48 + e6cb7ca
            for the full trace.
            """
            if getattr(self, "_channel_toggle_in_flight", False):
                print(f"[§2.5-step2] reentrant _toggle_all_labels DROPPED")
                return
            self._channel_toggle_in_flight = True
            try:
                all_layers = []
                for wl_name in all_wl_names:
                    ml = self._layer_by_name(self.viewer.layers, wl_name)
                    if ml is not None:
                        all_layers.append(ml)
                    for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
                        orient = self._orientation_for_vm(vm)
                        sl = self._layer_by_name(vm.layers, f"{wl_name}_{orient}")
                        if sl is not None:
                            all_layers.append(sl)

                if not all_layers:
                    return

                new_vis = not all_layers[0].visible

                if not new_vis:
                    for lyr in all_layers:
                        try:
                            lyr._epicycle_pre_hide_mode = str(lyr.mode)
                        except Exception:
                            pass

                for lyr in all_layers:
                    try:
                        lyr.visible = new_vis
                    except Exception:
                        pass

                if new_vis:
                    for lyr in all_layers:
                        try:
                            saved = getattr(lyr, "_epicycle_pre_hide_mode", None)
                            if saved and getattr(lyr, "editable", False) and hasattr(lyr, "mode"):
                                lyr.mode = saved
                        except Exception:
                            pass
            finally:
                self._channel_toggle_in_flight = False

        # Re-bind the labels key on all viewers with the multi-layer toggle.
        viewers = [self.viewer, self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]
        for v in viewers:
            try:
                v.bind_key(labels_key, _toggle_all_labels, overwrite=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Drawing constraints
    # ------------------------------------------------------------------

    def _enforce_brush_size_1(self):
        """Lock brush size to 1 on all label layers (main + subviewers)."""
        for layer in self._all_subviewer_label_layers():
            try:
                layer.brush_size = 1
            except Exception:
                pass
        # Also set on main viewer label layers.
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                try:
                    layer.brush_size = 1
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Lifecycle — cleanup
    # ------------------------------------------------------------------

    def setParent(self, parent):
        """Disconnect handlers before the base class destroys child Qt objects.

        DO NOT call _close_subviewers() here.  The base class setParent(None)
        Phase 1-3 (hide children → null refs → sendPostedEvents) destroys
        children safely.  Calling _close_subviewers() here would trigger an
        EXTRA sendPostedEvents flush (line 1002 in base _close_subviewers),
        then Phase 3 does ANOTHER flush — double destruction in the same
        call stack crashes vispy.  The 3D annotator survives because it does
        NOT call _close_subviewers inside setParent — it lets Phase 1-3
        handle destruction, and _close_subviewers runs later from the
        debounce hook (separate event tick, wrapped in try/except).
        """
        _diag("setParent", f"parent={'None' if parent is None else type(parent).__name__} | "
              f"cleanup_done={getattr(self, '_close_cleanup_done', '?')} | "
              f"canvases: {_canvas_status(self)}", stack=True)
        if parent is None:
            if not getattr(self, "_close_cleanup_done", False):
                self._close_cleanup()
            # Let base class Phase 1-3 handle child destruction.
            # Do NOT call _close_subviewers() here — see docstring.
        super().setParent(parent)
        if parent is None:
            _diag("setParent", f"AFTER super() | canvases: {_canvas_status(self)}")

    def _close_subviewers(self) -> None:
        """No-op override: let base class setParent Phase 1-3 handle destruction.

        The base class _close_subviewers() calls sendPostedEvents(DeferredDelete)
        which immediately destroys subviewer canvas C++ objects.  These share an
        OpenGL context with the main viewer.  Destroying them corrupts GL state
        and crashes napari on the next render (layer removal, patch load, etc.).

        The 3D annotator can call _close_subviewers() safely because its close
        hook fires in a separate event tick from setParent.  The 2D annotator
        cannot — both dock.close() (step exit) and setParent(None) (user close)
        trigger _close_subviewers in the same call stack as subsequent operations.

        Instead, we let napari's setParent(None) → base class Phase 1-3 destroy
        child widgets naturally (hide → deleteLater → flush).  This is safe
        because Phase 1-3 runs AFTER all our code completes.
        """
        _diag("_close_subviewers", "NO-OP — letting base class setParent handle destruction")
        self._subviewers_closed = True

    def _close_cleanup(self) -> None:
        """Teardown sequence for 2D HITL annotator."""
        _diag("_close_cleanup", f"ENTER | already_done={getattr(self, '_close_cleanup_done', '?')} | "
              f"canvases: {_canvas_status(self)}", stack=True)
        if getattr(self, "_close_cleanup_done", False):
            _diag("_close_cleanup", "BAIL — already done")
            return
        self._close_cleanup_done = True
        self._dock_close_hook_fired = True

        # Step 1: Set closing guard, invalidate deferred callbacks.
        self._is_closing = True
        self._invalidate_session_callbacks()

        # Step 2: Stop timers.
        for timer_attr in ("_init_auto_load_timer", "_deferred_auto_load_timer"):
            timer = getattr(self, timer_attr, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass

        # Step 3: Destroy satellite docks (Advanced Features, Multiviewer Settings).
        try:
            from napari_epicycle.annotator.launcher import _destroy_satellite_docks
            _destroy_satellite_docks(self.viewer)
        except Exception:
            pass

        # Step 4: Disconnect viewer-level event handlers.
        try:
            self._disconnect_viewer_events()
        except Exception:
            pass

        # Step 4b: Disconnect slider valueChanged handlers (stored in
        # _wire_signals override — base class uses anonymous lambdas
        # that can't be disconnected).
        for orient, (slider, cb) in list(getattr(self, "_slider_cbs", {}).items()):
            if slider is not None:
                try:
                    slider.valueChanged.disconnect(cb)
                except (ValueError, AttributeError, TypeError):
                    pass
        if hasattr(self, "_slider_cbs"):
            self._slider_cbs.clear()

        # Step 5: Disconnect layer insertion handler.
        try:
            self.viewer.layers.events.inserted.disconnect(self._layer_inserted_cb)
        except (ValueError, AttributeError):
            pass

        # Step 5b: Disconnect mode event listeners and teardown 2D hotkeys.
        self._disconnect_mode_events()
        self._teardown_2d_hotkeys()

        # Step 5b.1: Teardown per-subviewer focus filters (2026-05-26 redesign).
        try:
            self._teardown_focus_filters()
        except Exception:
            pass

        _diag("_close_cleanup", f"AFTER disconnects | viewer handlers: {_count_viewer_handlers(self.viewer)}")

        # Disconnect drawing callback.
        if hasattr(self, "drawing_handler") and self.drawing_handler is not None:
            self.drawing_handler.post_sync_callback = None

        # Step 6: Null out callback references to prevent stale calls.
        for attr in (
            "_segment_slice_callback",
            "_overlay_slice_callback",
            "_accept_slice_callback",
            "_patch_switch_callback",
            "_on_slice_edited_callback",
        ):
            try:
                setattr(self, attr, None)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Per-subviewer focus-driven layer activation (2026-05-26)
    #
    # Sidesteps the broken sync_active_layer cross-class cascade.  When
    # user clicks/interacts with a subviewer canvas, we route THAT VM's
    # selection.active to the class-matching labels layer for that
    # orientation (no cross-VM propagation).  Mode defaults to pan_zoom;
    # user opts into paint via napari's '2' keybind.  Per-view indicator
    # (bright amber glow) shows which view is the current paint surface.
    # ------------------------------------------------------------------

    def _setup_focus_filters(self) -> None:
        """Install per-subviewer mouse-press focus listeners.

        Mirrors _setup_2d_hotkeys' canvas-attribute defensive pattern:
        each canvas stores its filter via _epicycle_focus_filter so
        accidental re-setup can detect + clean prior filters before
        installing new ones.
        """
        self._teardown_focus_filters()
        self._focus_filters = []
        for orient, qt_viewer in [
            ("XY", self.qt_viewer_xy),
            ("XZ", self.qt_viewer_xz),
            ("YZ", self.qt_viewer_yz),
        ]:
            try:
                canvas = qt_viewer.canvas.native
                prior = getattr(canvas, "_epicycle_focus_filter", None)
                if prior is not None:
                    try:
                        canvas.removeEventFilter(prior)
                    except Exception:
                        pass
                    try:
                        delattr(canvas, "_epicycle_focus_filter")
                    except Exception:
                        pass
                filt = _SubviewerFocusFilter(self, orient, parent=canvas)
                canvas.installEventFilter(filt)
                try:
                    canvas._epicycle_focus_filter = filt
                except Exception:
                    pass
                self._focus_filters.append((canvas, filt))
            except Exception as exc:
                _diag("_setup_focus_filters", f"FAILED on {orient}: {exc}")

    def _teardown_focus_filters(self) -> None:
        """Remove focus filters from subviewer canvases."""
        for canvas, filt in getattr(self, "_focus_filters", []):
            try:
                canvas.removeEventFilter(filt)
            except Exception:
                pass
            try:
                delattr(canvas, "_epicycle_focus_filter")
            except Exception:
                pass
        self._focus_filters = []

    def _on_subviewer_focused(self, orient: str) -> None:
        """Route the focused subviewer's VM to the class-matching labels layer.

        Fires on EVERY mouse press on a subviewer canvas (filter granularity).
        To avoid killing mid-stroke paint, mode is only reset to pan_zoom on
        ORIENT SWITCH — same-orient clicks (including the mouse-down that
        starts a paint stroke) preserve the layer's current mode.

        Skips when the focused orientation is disabled for the current bucket
        mode (e.g. clicking XY in axial mode is ignored).
        """
        orient = str(orient).upper()
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        # Skip if this orientation isn't valid for current bucket.
        if target == "lateral" and orient != "XY":
            return
        if target == "axial" and orient == "XY":
            return

        # Only do "switch" work if the focused orient actually changed.
        # First focus (previous None) AND same-orient re-click both fall
        # through without touching layer.mode — critical to avoid resetting
        # paint mode on every mouse-down within a single subviewer.
        previous_focus = getattr(self, "_focused_orient", None)
        is_orient_switch = (previous_focus is not None and previous_focus != orient)

        vm = self._vm_for_orient(orient)
        if vm is None:
            return
        # Class-matching layer lookup — mirrors _set_subviewer_active_labels
        # logic but scoped to ONE VM (this orient only).
        patch = self._labels_patch_prefix(self._active_labels_name)
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
                # Only reassign if VM's active isn't already correct.  Avoids
                # spurious selection.events.active emissions on every click.
                if vm.layers.selection.active is not layer:
                    vm.layers.selection.active = layer
                # Reset mode ONLY on orient switch — not on first focus, not
                # on same-orient re-clicks.  User opts into paint via '2'
                # after switching; within a view, paint stays paint.
                if is_orient_switch:
                    layer.mode = "pan_zoom"
            except Exception:
                pass

        self._focused_orient = orient
        self._update_per_view_indicator()

    def _edit_state_for_orient(self, orient: str) -> str:
        """Edit-states (2026-05-27): map the currently-displayed slice of
        `orient`'s subviewer to an edit-state for the soft visual lock.

        Returns "editable" / "accepted" / "finalized".  Reads the status
        via the controller-injected `_slice_status_callback(view, idx)`.
        Defaults to "editable" whenever the callback is absent or the
        lookup fails — fail-open so the buttons never get stuck locked
        if the wiring isn't present.
        """
        cb = getattr(self, "_slice_status_callback", None)
        if cb is None:
            return "editable"
        vm = self._vm_for_orient(orient)
        if vm is None:
            return "editable"
        try:
            idx = int(vm.dims.current_step[0])
        except Exception:
            return "editable"
        try:
            status = str(cb(orient, idx) or "")
        except Exception:
            return "editable"
        if status == "accepted":
            return "accepted"
        if status in ("staged", "committed"):
            return "finalized"
        return "editable"

    def refresh_edit_state_visuals(self) -> None:
        """Public entry point (called by the controller after status
        transitions it makes that the annotator can't observe directly:
        demote-on-draw, Revert, accept, stage).  Re-runs the per-view
        button indicator AND re-enforces the hard-lock editability (so an
        Accept locks the slice immediately, a Revert unlocks it)."""
        try:
            self._update_per_view_indicator()
        except Exception as exc:
            _diag("refresh_edit_state_visuals", f"failed: {exc}")
        try:
            self._enforce_finalized_slice_lock()
        except Exception as exc:
            _diag("refresh_edit_state_visuals", f"enforce failed: {exc}")

    def _sync_slider_from_dims(self, orientation: str) -> None:
        """Override: after the base syncs the slider on a slice scroll,
        re-evaluate (a) the per-orient soft-lock button visuals and (b) the
        HARD-LOCK editability for the newly-displayed slice — locking the
        layer if you scrolled onto a finalized slice, re-enabling it if you
        scrolled onto an editable one (2026-05-27 edit-states + 2026-05-28
        hard-lock).  Cheap — status-dict lookups + a couple of property
        sets; no recompute or re-render."""
        super()._sync_slider_from_dims(orientation)
        try:
            self._update_per_view_indicator()
        except Exception:
            pass
        try:
            self._enforce_finalized_slice_lock()
        except Exception:
            pass

    def _update_per_view_indicator(self) -> None:
        """Single source of truth for the 6 per-orient Segment/Accept
        buttons' ENABLE state + STYLE.

        Factors three things:
          1. **Bucket** — only the active bucket's buttons are usable
             (lateral → XY; axial → XZ+YZ).  Was previously set inline in
             `_refresh_segment_button_text`; consolidated here 2026-05-27
             so bucket + edit-state can't fight over the enabled flag.
          2. **Edit-state** (2026-05-27) — if the displayed slice for an
             orient is accepted/finalized, its buttons are disabled +
             tinted emerald/ice (soft lock; a deliberate paint still
             demotes the slice back to curating via the controller).
          3. **Focus + paint mode** — the focused orient's buttons glow
             when its labels layer is in paint mode (editable slices only).

        Enabled = bucket_active AND edit_state == "editable".
        """
        target = str(getattr(self, "_active_target_mode", "lateral")).lower()
        focused = getattr(self, "_focused_orient", None)
        in_paint = False
        if focused:
            try:
                vm = self._vm_for_orient(focused)
                if vm is not None:
                    active = vm.layers.selection.active
                    if active is not None and getattr(active, "mode", None) == "paint":
                        in_paint = True
            except Exception:
                pass
        button_pairs = [
            ("XY", getattr(self, "segment_mode_btn", None), getattr(self, "accept_slice_btn", None)),
            ("XZ", getattr(self, "segment_xz_btn", None), getattr(self, "accept_xz_btn", None)),
            ("YZ", getattr(self, "segment_yz_btn", None), getattr(self, "accept_yz_btn", None)),
        ]
        for orient, seg_btn, acc_btn in button_pairs:
            bucket_active = (
                (orient == "XY" and target != "axial")
                or (orient in {"XZ", "YZ"} and target == "axial")
            )
            edit_state = self._edit_state_for_orient(orient)
            enabled = bucket_active and (edit_state == "editable")
            is_active_paint = (orient == focused) and in_paint and enabled
            style = _orient_button_style(
                orient, paint_active=is_active_paint, edit_state=edit_state
            )
            for btn in (seg_btn, acc_btn):
                if btn is None:
                    continue
                try:
                    btn.setEnabled(enabled)
                    btn.setStyleSheet(style)
                except Exception:
                    pass

    def closeEvent(self, event) -> None:
        """Clean up on dock close."""
        _diag("closeEvent", f"FIRED | canvases: {_canvas_status(self)}")
        self._close_cleanup()
        self._close_subviewers()
        try:
            from napari_epicycle.annotator.launcher import (
                _schedule_close_orphan_sweep_retries,
            )
            _schedule_close_orphan_sweep_retries()
        except Exception:
            pass
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        """Diagnostic: trace when the inner widget is hidden."""
        _diag("hideEvent", f"FIRED | canvases: {_canvas_status(self)}")
        super().hideEvent(event)
