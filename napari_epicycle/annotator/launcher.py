"""Launcher for the native EpiCycle multiview annotator and advanced features docks."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication, QDockWidget, QFileDialog


def _curation_debug_enabled() -> bool:
    import os
    return os.environ.get("EPICYCLE_CURATION_DEBUG") == "1"


def _curation_debug(msg: str) -> None:
    if _curation_debug_enabled():
        print(f"[CURATION-DEBUG] {msg}")


def _qt_main_window(viewer):
    window = getattr(viewer, "window", None)
    if window is None:
        return None
    return getattr(window, "_qt_window", None)


def _qobj_deleted(obj) -> bool:
    """True when ``obj`` is a Qt object whose C++ side is gone.

    Anything that is not a sip wrapper (a test fake, a plain object) is
    answered ``False``.
    """
    if obj is None:
        return False
    try:
        from qtpy import sip
    except Exception:
        try:
            import sip  # type: ignore
        except Exception:
            return False
    try:
        return bool(sip.isdeleted(obj))
    except Exception:
        return False


def _dock_is_empty(dock) -> bool:
    """Is ``dock`` a dock that holds nothing any more?

    napari's title-bar X can take the inner widget out of a dock (and out of
    napari's registry) while the ``QDockWidget`` itself stays attached to the
    main window with its title.  Such a dock -- no widget, a deleted widget,
    or itself deleted -- must never be "reused" (live driver, part G / F5).
    A dock object with no ``widget()`` at all (a test fake that stands for
    its own widget) is not empty.
    """
    if dock is None:
        return False
    if _qobj_deleted(dock):
        return True
    if not hasattr(dock, "widget"):
        return False
    try:
        inner = dock.widget()
    except Exception:
        return True
    return inner is None or _qobj_deleted(inner)


def _find_dock_by_title(viewer, title: str):
    from qtpy.QtWidgets import QDockWidget

    qt_main = _qt_main_window(viewer)
    if qt_main is None:
        return None
    try:
        for dock in qt_main.findChildren(QDockWidget):
            if str(dock.windowTitle()) == str(title):
                # An empty dock left behind by napari's X is not the dock
                # this title names any more; keep looking.
                if _dock_is_empty(dock):
                    continue
                return dock
    except Exception:
        return None
    return None


def _purge_empty_docks(viewer, title: str, *, accept=None) -> int:
    """Remove every dock titled ``title`` that holds nothing we can reuse.

    A dock is purged when it is empty (``_dock_is_empty``) or, when
    ``accept`` is given, when its widget fails ``accept(widget)``.  Each one
    is hidden first (so its visual leaves the window before it is freed) and
    then removed through ``_destroy_dock``.  Returns how many were purged.
    Called by the launchers only -- never from a hot getter, because
    ``_destroy_dock`` flushes deferred deletes.
    """
    from qtpy.QtWidgets import QDockWidget

    qt_main = _qt_main_window(viewer)
    if qt_main is None:
        return 0
    try:
        docks = [
            d for d in qt_main.findChildren(QDockWidget)
            if not _qobj_deleted(d) and str(d.windowTitle()) == str(title)
        ]
    except Exception:
        return 0
    purged = 0
    for dock in docks:
        stale = _dock_is_empty(dock)
        if not stale and accept is not None:
            try:
                stale = not accept(dock.widget())
            except Exception:
                stale = True
        if not stale:
            continue
        print(f"[DOCK-REUSE] purging a stale {title!r} dock (empty or foreign)")
        try:
            dock.hide()
        except Exception:
            pass
        _destroy_dock(viewer, dock)
        if not _qobj_deleted(dock):
            # A deferred delete napari's X posted at an OUTER scope (the
            # top of the event loop) is not flushed from a deeper one (a
            # worker's finished signal): the dock outlived the purge, kept
            # its title, and was purged again on every launch (Step 4 W6,
            # 2026-10-05).  Off the window already; untitled, it is never
            # found again, and the event loop deletes it.
            try:
                dock.setWindowTitle("")
                dock.setObjectName("")
            except Exception:
                pass
        purged += 1
    return purged


def _resolve_existing_dock(viewer, dock_name: str):
    # Prefer true QDockWidget lookup by window title.
    dock = _find_dock_by_title(viewer, dock_name)
    if dock is not None:
        return dock

    window = getattr(viewer, "window", None)
    if window is None:
        return None
    registry = getattr(window, "dock_widgets", None)
    if registry is None:
        registry = getattr(window, "_dock_widgets", None)
    if isinstance(registry, dict):
        dock = registry.get(dock_name)
        if dock is not None and not _dock_is_empty(dock):
            return dock
    return None


def _find_all_docks_by_title(viewer, title: str):
    """Return all dock widgets whose window title matches ``title``."""
    from qtpy.QtWidgets import QDockWidget

    qt_main = _qt_main_window(viewer)
    results = []
    if qt_main is not None:
        try:
            for dock in qt_main.findChildren(QDockWidget):
                if str(dock.windowTitle()) == str(title):
                    results.append(dock)
        except Exception:
            pass

    window = getattr(viewer, "window", None)
    registry = getattr(window, "dock_widgets", None) if window is not None else None
    if registry is None and window is not None:
        registry = getattr(window, "_dock_widgets", None)
    if isinstance(registry, dict):
        dock = registry.get(title)
        if dock is not None and all(id(d) != id(dock) for d in results):
            results.append(dock)
    return results


def _run_widget_cleanup(widget) -> None:
    """Run close cleanup methods on an annotator widget.

    After cleanup, the widget is hidden and scheduled for deletion to prevent
    it from persisting as a top-level zombie window.  napari's
    ``remove_dock_widget`` calls ``widget.setParent(None)`` which orphans the
    inner widget without closing or deleting it.  Without explicit deletion
    here, the widget accumulates as an invisible top-level window across
    close/relaunch cycles, causing progressive sluggishness.

    **Critical:** After napari orphans the annotator via ``setParent(None)``,
    the annotator's child QWidgets (QPushButtons, QCheckBoxes, QGroupBoxes,
    etc.) can also become top-level orphans.  Windows displays them on the
    taskbar using the QApplication name ("magicgui") because they have no
    windowTitle.  We must hide ALL child widgets before napari can orphan them,
    and then schedule them for deletion to prevent accumulation.
    """
    if widget is None:
        return
    # Forget the annotator's cached Advanced Features dock (final fix pass P1).
    try:
        widget._afw_cache = None
    except Exception:
        pass
    if hasattr(widget, "_close_cleanup"):
        try:
            widget._close_cleanup()
        except Exception:
            pass
    if hasattr(widget, "_close_subviewers"):
        try:
            widget._close_subviewers()
        except Exception:
            pass
    # Child widget orphan prevention is handled by the setParent(None)
    # override on NativeMultiviewAnnotator — it hides + deleteLater() all
    # children at the exact moment napari unparents the widget.
    #
    # Hide the widget itself to remove it from the Windows taskbar.
    # Do NOT call widget.close() — that triggers closeEvent which would
    # double-run _close_cleanup.
    try:
        widget.hide()
    except Exception:
        pass
    # **DO NOT** call widget.deleteLater() here.  napari's destroyOnClose
    # calls setParent(None) AFTER this hook.  Our setParent override flushes
    # deferred deletes via sendPostedEvents(DeferredDelete).  If this hook
    # already queued the widget for deletion, that flush destroys the widget
    # MID-EXECUTION of setParent, orphaning all children.  The widget will
    # be cleaned up by _sweep_orphan_annotator_widgets on the next launch.
    _schedule_close_orphan_sweep_retries()


def _hide_and_destroy_child_widgets(widget) -> None:
    """Hide and schedule deletion of all child QWidgets.

    Prevents annotator children (QPushButton, QCheckBox, QGroupBox, etc.)
    from becoming visible top-level orphans when napari's
    ``remove_dock_widget`` calls ``setParent(None)`` on the annotator.
    """
    try:
        from qtpy.QtWidgets import QWidget as _QW
    except Exception:
        return
    try:
        children = widget.findChildren(_QW)
    except Exception:
        return
    for child in children:
        try:
            child.hide()
        except Exception:
            pass
        try:
            child.deleteLater()
        except Exception:
            pass


def _cleanup_old_annotator_widget(widget) -> None:
    """Call cleanup methods on an annotator widget before its dock is destroyed.

    ``_destroy_dock()`` does NOT trigger ``closeEvent()`` on the inner widget,
    so cleanup that normally runs in ``closeEvent → _close_cleanup`` is skipped.
    This helper explicitly invokes the same cleanup methods so guard flags are
    reset, event handlers are disconnected, navigator callbacks are cleared,
    and QtViewer child windows are closed — preventing stale references and
    magicgui window accumulation on relaunch.

    Tolerates missing methods and exceptions so it never blocks dock destruction.
    """
    if widget is None:
        return
    # Idempotent guard: if close hook already ran, skip duplicate cleanup.
    if getattr(widget, "_dock_close_hook_fired", False):
        return
    widget._dock_close_hook_fired = True
    _run_widget_cleanup(widget)


def _install_dock_close_hook_if_alive(dock, inner_widget) -> None:
    """Deferred-timer-safe wrapper around :func:`_install_dock_close_hook`.

    Journey #17 (2026-09-07): the 2D launcher defers hook installation by
    600 ms so napari's initial tabification doesn't trip the visibility
    debounce.  A user who closes the annotator inside that window leaves the
    timer holding a dock whose C++ side is already gone, and touching it
    raises (or, worse, crashes).  Check first, and never let this path take
    the session down.
    """
    if dock is None or inner_widget is None:
        return
    # `import sip` raises ModuleNotFoundError on PyQt6 (it ships as
    # PyQt6.sip), which made this guard dead code.  qtpy re-exports the right
    # one for the active binding — same pattern as
    # `napari_epicycle/utils/crash_logger.py::liveness`.
    try:
        from qtpy import sip  # type: ignore
    except Exception:
        sip = None
    if sip is not None:
        try:
            if sip.isdeleted(dock) or sip.isdeleted(inner_widget):
                print("[LAUNCH] deferred close-hook skipped: dock already "
                      "destroyed")
                return
        except (TypeError, RuntimeError):
            # Not a wrapped C++ object (a test double, PySide) — fall through
            # to the guarded call below.
            pass
    try:
        _install_dock_close_hook(dock, inner_widget)
    except RuntimeError as exc:
        print(f"[LAUNCH] deferred close-hook skipped: {exc}")


def _install_dock_close_hook(dock, inner_widget) -> None:
    """Install close-event hook so cleanup runs when dock is closed.

    Uses QEvent.Close (eventFilter) instead of visibilityChanged to avoid
    firing cleanup on tab switches or transient hide/show transitions.
    """
    if dock is None or inner_widget is None:
        return
    if getattr(inner_widget, "_dock_close_hook_installed", False):
        return

    try:
        from qtpy.QtCore import QObject, QEvent
    except Exception:
        return

    def _mark_and_cleanup() -> None:
        if getattr(inner_widget, "_dock_close_hook_fired", False):
            return
        inner_widget._dock_close_hook_fired = True
        _run_widget_cleanup(inner_widget)

    def _should_cleanup_hidden_dock() -> bool:
        """Return True when hidden state likely came from user close."""
        # Guard: if the main window is minimized, the dock becomes invisible
        # but the user did NOT close the annotator.  Never clean up in this case.
        try:
            main_win = _qt_main_window(inner_widget.viewer)
            if main_win is not None and main_win.isMinimized():
                return False
        except Exception:
            pass
        # Preferred signal: QDockWidget's toggle action is unchecked on close.
        try:
            if hasattr(dock, "toggleViewAction"):
                action = dock.toggleViewAction()
                if action is not None and hasattr(action, "isChecked"):
                    return not bool(action.isChecked())
        except Exception:
            pass
        # Fallback if toggle action is unavailable.
        try:
            return not bool(dock.isVisible())
        except Exception:
            return True

    class _DockCloseCleanupFilter(QObject):
        def __init__(self, target_widget):
            super().__init__()
            self._target_widget = target_widget

        def eventFilter(self, obj, event):
            try:
                et = event.type()
            except Exception:
                return False
            is_close = et == QEvent.Close
            is_hide = et == QEvent.Hide
            if not is_close and not is_hide:
                return False
            # 2026-09-07 (console-encoding hazard): the cleanup runs
            # FIRST and the diagnostic second.  A print can RAISE
            # (UnicodeEncodeError on a non-UTF-8 Windows console is a
            # ValueError subclass); printing first aborted the dock-close
            # cleanup entirely.  Nothing between the branch decision and
            # _mark_and_cleanup() may raise.
            if is_close:
                _mark_and_cleanup()
                print("[DOCK-FILTER] QEvent.Close -> cleanup done")
            elif is_hide and _should_cleanup_hidden_dock():
                _mark_and_cleanup()
                print("[DOCK-FILTER] QEvent.Hide + toggle unchecked -> cleanup done")
            return False

    try:
        filt = _DockCloseCleanupFilter(inner_widget)
        dock.installEventFilter(filt)
        # Also listen to visibilityChanged: some dock-close paths emit hide
        # without a close event depending on host docking behavior.
        vc_cb = None
        if hasattr(dock, "visibilityChanged"):
            try:
                from qtpy.QtCore import QTimer

                _vc_debounce = QTimer()
                _vc_debounce.setSingleShot(True)
                _vc_debounce.setInterval(300)

                def _vc_debounce_check():
                    """Run cleanup only if dock is still hidden after debounce."""
                    should = _should_cleanup_hidden_dock()
                    print(f"[VIS-DEBOUNCE] timer fired, should_cleanup={should}")
                    if should:
                        _mark_and_cleanup()

                _vc_debounce.timeout.connect(_vc_debounce_check)

                def _on_visibility_changed(visible):
                    if visible:
                        # Dock became visible again — cancel pending cleanup.
                        was_active = _vc_debounce.isActive()
                        _vc_debounce.stop()
                        if was_active:
                            print(f"[VIS-DEBOUNCE] visible=True, cancelled pending cleanup")
                        return
                    # If the main window is minimized, this is NOT a dock close.
                    # Skip scheduling cleanup entirely.
                    try:
                        main_win = _qt_main_window(inner_widget.viewer)
                        if main_win is not None and main_win.isMinimized():
                            print(f"[VIS-DEBOUNCE] visible=False but window minimized, ignoring")
                            return
                    except Exception:
                        pass
                    # Debounce: transient hides from napari layout updates
                    # (layer add/remove) resolve within ~50ms.  Only run
                    # cleanup if the dock stays hidden for 300ms.
                    print(f"[VIS-DEBOUNCE] visible=False, scheduling 300ms debounce")
                    _vc_debounce.start()

                dock.visibilityChanged.connect(_on_visibility_changed)
                vc_cb = _on_visibility_changed
                # Keep timer alive.
                inner_widget._dock_close_vc_debounce = _vc_debounce
            except Exception:
                vc_cb = None
        # Keep strong refs on widget to avoid GC of event filter.
        inner_widget._dock_close_hook_filter = filt
        inner_widget._dock_close_visibility_cb = vc_cb
        inner_widget._dock_close_hook_installed = True
    except Exception:
        pass


def _destroy_dock(viewer, dock):
    """Fully remove a dock widget and flush Qt events so the layout is clean."""
    from qtpy.QtCore import QCoreApplication, QEvent

    try:
        window = getattr(viewer, "window", None)
        widget = dock.widget() if hasattr(dock, "widget") else None
        if window is not None and hasattr(window, "remove_dock_widget"):
            # Remove from napari before closing children. Avoid setParent(None):
            # unparenting can orphan transient top-level "magicgui" windows.
            try:
                window.remove_dock_widget(dock)
            except Exception:
                try:
                    window.remove_dock_widget(widget if widget is not None else dock)
                except Exception:
                    pass
        if widget is not None:
            try:
                widget.close()
            except Exception:
                pass
            try:
                widget.deleteLater()
            except Exception:
                pass
        try:
            dock.close()
        except Exception:
            pass
        try:
            dock.deleteLater()
        except Exception:
            pass
    except Exception:
        pass
    try:
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    except Exception:
        pass


def _close_orphan_magicgui_windows_global() -> None:
    """Close detached top-level windows titled/classed as magicgui."""
    try:
        from qtpy.QtCore import QCoreApplication, QEvent
        from qtpy.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    try:
        top_levels = list(app.topLevelWidgets())
    except Exception:
        return
    closed_count = 0
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
        # Match magicgui windows AND orphan QtPopup/QtModalPopup windows
        # (created by napari dims play buttons, survive QtViewer close).
        is_orphan = (
            "magicgui" in title or "magicgui" in cls or "magicgui" in obj_name
            or "qtmodalpopup" in obj_name or "qtpopup" in cls
        )
        if not is_orphan:
            continue
        closed_count += 1
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
        # Only flush deferred deletes — do NOT call processEvents() as it
        # can materialize new magicgui windows from pending events.
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    except Exception:
        pass
    pass  # sweep complete


def _schedule_close_orphan_sweep_retries() -> None:
    """Schedule retry orphan sweeps for windows created late in close."""
    try:
        from qtpy.QtCore import QTimer
    except Exception:
        _close_orphan_magicgui_windows_global()
        return
    _close_orphan_magicgui_windows_global()
    for delay_ms in (0, 50, 125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000):
        try:
            QTimer.singleShot(delay_ms, _close_orphan_magicgui_windows_global)
        except Exception:
            pass


def _sweep_orphan_annotator_widgets() -> None:
    """Destroy orphan annotator widgets that escaped dock close cleanup.

    napari's ``remove_dock_widget`` calls ``widget.setParent(None)`` which
    orphans the inner widget without closing or deleting it.  If the close
    hook's ``deleteLater()`` call didn't take effect (e.g., no event loop
    tick between close and relaunch), the widget persists as a parentless
    top-level zombie.

    This sweep finds any ``NativeMultiviewAnnotator`` or
    ``Curation3DAnnotator`` instances in ``topLevelWidgets()`` that have
    no parent (i.e., zombies) and deletes them.  Called at the start of
    launch functions to clean up before creating a new widget.
    """
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    _ANNOTATOR_CLASSES = ("NativeMultiviewAnnotator", "Curation3DAnnotator", "Curation2DAnnotator")
    found_zombies = []
    for w in list(app.topLevelWidgets()):
        if w is None:
            continue
        cls_name = type(w).__name__
        if cls_name not in _ANNOTATOR_CLASSES:
            continue
        # Only destroy parentless zombies — a properly docked widget has a parent.
        try:
            if w.parent() is not None:
                continue
        except Exception:
            pass
        found_zombies.append((cls_name, w))
        print(f"[SWEEP-DIAG] found zombie: {cls_name} cleanup_done={getattr(w, '_close_cleanup_done', '?')} hook_fired={getattr(w, '_dock_close_hook_fired', '?')}")
        # Run cleanup if not already done.
        if not getattr(w, "_close_cleanup_done", False):
            print(f"[SWEEP-DIAG]   running _run_widget_cleanup on {cls_name}")
            _run_widget_cleanup(w)
        else:
            # Already cleaned up — just hide and delete.
            print(f"[SWEEP-DIAG]   already cleaned, calling hide+deleteLater on {cls_name}")
            try:
                w.hide()
            except Exception:
                pass
            try:
                w.deleteLater()
            except Exception:
                pass
    if found_zombies:
        print(f"[SWEEP-DIAG] flushing deferred deletes ({len(found_zombies)} zombies found)")
    else:
        print(f"[SWEEP-DIAG] no zombies found")
    try:
        from qtpy.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    except Exception:
        pass
    if found_zombies:
        print(f"[SWEEP-DIAG] flush complete")


def _destroy_satellite_docks(viewer) -> None:
    """Destroy all known satellite docks of the 3D curation annotator.

    Satellite docks are floating/docked widgets launched by the annotator
    (Advanced Features, Multiviewer Settings, Label Navigator) that can
    survive annotator close and accumulate across relaunch cycles.
    """
    satellite_names = [
        _ADVANCED_DOCK_NAMES.get("3d_curation", "EpiCycle 3D Advanced Features"),
        _MULTIVIEWER_SETTINGS_DOCK_NAME,
        _NAVIGATOR_DOCK_NAME,
    ]
    for name in satellite_names:
        for dock in _find_all_docks_by_title(viewer, name):
            try:
                inner = dock.widget() if hasattr(dock, "widget") else None
                if inner is not None:
                    inner.close()
            except Exception:
                pass
            _destroy_dock(viewer, dock)


def _find_epicycle_main_dock(viewer, *, exclude_titles: set[str]):
    from qtpy.QtWidgets import QDockWidget

    qt_main = _qt_main_window(viewer)
    if qt_main is None:
        return None
    try:
        docks = list(qt_main.findChildren(QDockWidget))
    except Exception:
        return None
    def _score(dock: QDockWidget) -> tuple[int, int, int]:
        """Prefer visible, non-floating plugin host docks."""
        try:
            visible = 1 if dock.isVisible() else 0
        except Exception:
            visible = 0
        try:
            floating = 0 if dock.isFloating() else 1
        except Exception:
            floating = 0
        area = 0
        try:
            widget = dock.widget()
            if widget is not None:
                name = widget.__class__.__name__
                if name in {"EpiCycleMainWidget", "EpiCycleWorkflowShell"}:
                    area = 1
        except Exception:
            area = 0
        return (area, visible, floating)

    candidates: list[QDockWidget] = []
    for dock in docks:
        title = str(dock.windowTitle() or "")
        if title in exclude_titles:
            continue
        if "EpiCycle Plugin" in title:
            candidates.append(dock)
    if not candidates:
        return None
    return sorted(candidates, key=_score, reverse=True)[0]


def _tabify_native_with_epicycle(
    viewer,
    native_dock,
    native_dock_name: str,
    *,
    retries: int = 0,
) -> None:
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QDockWidget

    qt_main = _qt_main_window(viewer)
    if qt_main is None:
        return
    if not isinstance(native_dock, QDockWidget):
        native_dock = _find_dock_by_title(viewer, native_dock_name)
    if native_dock is None:
        if retries > 0:
            QTimer.singleShot(
                80,
                lambda: _tabify_native_with_epicycle(
                    viewer,
                    None,
                    native_dock_name,
                    retries=retries - 1,
                ),
            )
        return
    host = _find_epicycle_main_dock(
        viewer,
        exclude_titles={
            native_dock_name,
            "EpiCycle Workflow Sidebar",
            "EpiCycle Run + Artifact History",
        },
    )
    if host is None or host is native_dock:
        if retries > 0:
            QTimer.singleShot(
                80,
                lambda: _tabify_native_with_epicycle(
                    viewer,
                    native_dock,
                    native_dock_name,
                    retries=retries - 1,
                ),
            )
        return
    try:
        qt_main.tabifyDockWidget(host, native_dock)
        native_dock.show()
        if hasattr(native_dock, "raise_"):
            native_dock.raise_()
    except Exception:
        return


def launch_native_multiview_annotator(
    viewer,
    *,
    dock_name: str = "EpiCycle 2D Training Annotator",
    image_layer_name: str = "",
    labels_layer_name: str = "",
    patch_ids=None,
    active_patch_id: str = "",
    target_view: str = "",
    selected_view: str = "",
    switch_patch_callback=None,
    segment_slice_callback=None,
    overlay_slice_callback=None,
    accept_slice_callback=None,
    navigate_to_slice_callback=None,
    enable_3d_curation: bool = False,
    flagged_label_ids: list = None,
    flagged_centroids: dict = None,
) -> Tuple[bool, str]:
    """Open native annotator dock and reuse existing dock when present."""
    if viewer is None:
        return False, "Viewer is not available."

    window = getattr(viewer, "window", None)
    if window is None:
        return False, "Viewer window is not available."

    _purge_empty_docks(viewer, dock_name)
    dock = _resolve_existing_dock(viewer, dock_name)
    if dock is not None:
        visible = False
        try:
            visible = dock.isVisible()
        except Exception:
            pass
        if visible:
            # Dock is already open and visible — update context and raise it.
            try:
                target_widget = dock.widget() if hasattr(dock, "widget") else dock
                if hasattr(target_widget, "set_patch_options"):
                    target_widget.set_patch_options(
                        patch_ids=patch_ids or [],
                        active_patch_id=active_patch_id,
                        switch_patch_callback=switch_patch_callback,
                    )
                if hasattr(target_widget, "set_runtime_callbacks"):
                    try:
                        target_widget.set_runtime_callbacks(
                            segment_slice_callback=segment_slice_callback,
                            overlay_slice_callback=overlay_slice_callback,
                            accept_slice_callback=accept_slice_callback,
                            navigate_to_slice_callback=navigate_to_slice_callback,
                        )
                    except TypeError:
                        try:
                            target_widget.set_runtime_callbacks(
                                segment_slice_callback=segment_slice_callback,
                                overlay_slice_callback=overlay_slice_callback,
                                accept_slice_callback=accept_slice_callback,
                            )
                        except TypeError:
                            try:
                                target_widget.set_runtime_callbacks(
                                    segment_slice_callback=segment_slice_callback,
                                    overlay_slice_callback=overlay_slice_callback,
                                )
                            except TypeError:
                                target_widget.set_runtime_callbacks(
                                    segment_slice_callback=segment_slice_callback,
                                )
                if hasattr(target_widget, "set_context"):
                    target_widget.set_context(
                        image_layer_name=image_layer_name,
                        labels_layer_name=labels_layer_name,
                        auto_load=True,
                        target_view=target_view,
                        selected_view=selected_view,
                    )
                if enable_3d_curation and hasattr(target_widget, "set_3d_curation_mode"):
                    target_widget.set_3d_curation_mode(True)
                if flagged_label_ids and hasattr(target_widget, "set_focused_refinement"):
                    target_widget.set_focused_refinement(
                        flagged_label_ids or [], flagged_centroids or {},
                    )
                if hasattr(dock, "raise_"):
                    dock.raise_()
                _tabify_native_with_epicycle(viewer, dock, dock_name, retries=3)
            except Exception:
                pass
            return True, "Native multiview annotator is already open."
        else:
            # Dock was closed/hidden — destroy it so we get a fresh layout.
            # Call cleanup explicitly — _destroy_dock does NOT trigger closeEvent.
            try:
                inner = dock.widget() if hasattr(dock, "widget") else dock
                _cleanup_old_annotator_widget(inner)
            except Exception:
                pass
            _destroy_dock(viewer, dock)

    try:
        module = importlib.import_module("napari_epicycle.annotator.native_multiview_widget")
        widget_cls = getattr(module, "NativeMultiviewAnnotator")
        widget = widget_cls(viewer)
        if hasattr(widget, "set_patch_options"):
            widget.set_patch_options(
                patch_ids=patch_ids or [],
                active_patch_id=active_patch_id,
                switch_patch_callback=switch_patch_callback,
            )
        if hasattr(widget, "set_runtime_callbacks"):
            try:
                widget.set_runtime_callbacks(
                    segment_slice_callback=segment_slice_callback,
                    overlay_slice_callback=overlay_slice_callback,
                    accept_slice_callback=accept_slice_callback,
                    navigate_to_slice_callback=navigate_to_slice_callback,
                )
            except TypeError:
                try:
                    widget.set_runtime_callbacks(
                        segment_slice_callback=segment_slice_callback,
                        overlay_slice_callback=overlay_slice_callback,
                        accept_slice_callback=accept_slice_callback,
                    )
                except TypeError:
                    try:
                        widget.set_runtime_callbacks(
                            segment_slice_callback=segment_slice_callback,
                            overlay_slice_callback=overlay_slice_callback,
                        )
                    except TypeError:
                        widget.set_runtime_callbacks(
                            segment_slice_callback=segment_slice_callback,
                        )
        if hasattr(widget, "set_context"):
            widget.set_context(
                image_layer_name=image_layer_name,
                labels_layer_name=labels_layer_name,
                auto_load=True,
                target_view=target_view,
                selected_view=selected_view,
            )
        if enable_3d_curation and hasattr(widget, "set_3d_curation_mode"):
            widget.set_3d_curation_mode(True)
        if flagged_label_ids and hasattr(widget, "set_focused_refinement"):
            widget.set_focused_refinement(
                flagged_label_ids or [], flagged_centroids or {},
            )
        window.add_dock_widget(widget, name=dock_name, area="right")
        dock = _find_dock_by_title(viewer, dock_name) or _resolve_existing_dock(viewer, dock_name)
        _tabify_native_with_epicycle(viewer, dock, dock_name, retries=6)
        return True, "Opened native multiview annotator."
    except Exception as exc:
        return False, f"Failed to open native multiview annotator: {exc}"


def launch_hitl_2d_annotator(
    viewer,
    *,
    dock_name: str = "EpiCycle 2D Training Annotator",
    image_layer_name: str = "",
    labels_layer_name: str = "",
    patch_ids=None,
    active_patch_id: str = "",
    target_view: str = "",
    selected_view: str = "",
    switch_patch_callback=None,
    segment_slice_callback=None,
    overlay_slice_callback=None,
    accept_slice_callback=None,
    navigate_to_slice_callback=None,
) -> Tuple[bool, str]:
    """Open the HITL 2D annotator dock (Curation2DAnnotator subclass).

    Follows the same lifecycle pattern as launch_3d_curation_annotator:
    orphan sweep, hidden dock cleanup, dock close hook installation.
    """
    if viewer is None:
        return False, "Viewer is not available."

    window = getattr(viewer, "window", None)
    if window is None:
        return False, "Viewer window is not available."

    # 2026-09-04: annotators need synchronous slicing (hide -> recolour ->
    # re-show within one tick).  Turn napari's persisted experimental.async_
    # OFF unless a Step-1 progressive loader is live on this viewer.
    try:
        from packages.epicycle_largedata.progressive import ensure_sync_slicing
        ensure_sync_slicing(viewer, reason="2D annotator launch")
    except Exception:
        pass

    def _wire_widget(target_widget):
        if hasattr(target_widget, "set_patch_options"):
            target_widget.set_patch_options(
                patch_ids=patch_ids or [],
                active_patch_id=active_patch_id,
                switch_patch_callback=switch_patch_callback,
            )
        if hasattr(target_widget, "set_runtime_callbacks"):
            try:
                target_widget.set_runtime_callbacks(
                    segment_slice_callback=segment_slice_callback,
                    overlay_slice_callback=overlay_slice_callback,
                    accept_slice_callback=accept_slice_callback,
                    navigate_to_slice_callback=navigate_to_slice_callback,
                )
            except TypeError:
                try:
                    target_widget.set_runtime_callbacks(
                        segment_slice_callback=segment_slice_callback,
                        overlay_slice_callback=overlay_slice_callback,
                        accept_slice_callback=accept_slice_callback,
                    )
                except TypeError:
                    pass
        if hasattr(target_widget, "set_context"):
            target_widget.set_context(
                image_layer_name=image_layer_name,
                labels_layer_name=labels_layer_name,
                auto_load=True,
                target_view=target_view,
                selected_view=selected_view,
            )

    # Sweep zombie widgets from previous close/relaunch cycles.
    _sweep_orphan_annotator_widgets()

    # Resolve all docks — purge hidden duplicates.
    _purge_empty_docks(viewer, dock_name)
    docks = _find_all_docks_by_title(viewer, dock_name)
    if not docks:
        dock = _resolve_existing_dock(viewer, dock_name)
        if dock is not None:
            docks = [dock]

    visible_dock = None
    hidden_docks = []
    for dock in docks:
        try:
            is_visible = bool(dock.isVisible())
        except Exception:
            is_visible = False
        if is_visible and visible_dock is None:
            visible_dock = dock
        else:
            hidden_docks.append(dock)

    # Always purge hidden duplicates first.
    for dock in hidden_docks:
        try:
            inner = dock.widget() if hasattr(dock, "widget") else dock
            _cleanup_old_annotator_widget(inner)
        except Exception:
            pass
        _destroy_dock(viewer, dock)

    if visible_dock is not None:
        try:
            target_widget = (
                visible_dock.widget()
                if hasattr(visible_dock, "widget")
                else visible_dock
            )
            # Reset close-hook guard so it fires on the NEXT close.
            # Without this, the hook from the previous close/relaunch
            # cycle stays fired=True and cleanup is skipped on next close
            # → handlers accumulate across relaunch cycles.
            target_widget._dock_close_hook_fired = False
            target_widget._close_cleanup_done = False
            target_widget._is_closing = False
            _install_dock_close_hook(visible_dock, target_widget)
            _wire_widget(target_widget)
            if hasattr(visible_dock, "raise_"):
                visible_dock.raise_()
        except Exception:
            pass
        return True, "HITL 2D annotator is already open."

    # No visible dock — destroy satellite docks + create fresh widget.
    _destroy_satellite_docks(viewer)
    _close_orphan_magicgui_windows_global()

    try:
        module = importlib.import_module(
            "napari_epicycle.annotator.curation_2d_widget"
        )
        widget_cls = getattr(module, "Curation2DAnnotator")
        widget = widget_cls(viewer)
        _wire_widget(widget)
        window.add_dock_widget(widget, name=dock_name, area="right")
        dock = (
            _find_dock_by_title(viewer, dock_name)
            or _resolve_existing_dock(viewer, dock_name)
        )
        _tabify_native_with_epicycle(viewer, dock, dock_name, retries=6)
        if dock is not None:
            # Defer close hook installation so the visibility debounce
            # doesn't fire during napari's initial tabification layout.
            # The 3D annotator survives this implicitly (its __init__ is
            # slow enough), but the 2D annotator's faster init means
            # the debounce fires before the dock is fully visible.
            from qtpy.QtCore import QTimer
            QTimer.singleShot(
                600,
                lambda d=dock, w=widget: _install_dock_close_hook_if_alive(d, w),
            )
        return True, "Opened HITL 2D annotator."
    except Exception as exc:
        return False, f"Failed to open HITL 2D annotator: {exc}"


def launch_3d_curation_annotator(
    viewer,
    *,
    dock_name: str = "EpiCycle 3D Curation Annotator",
    image_layer_name: str = "",
    labels_layer_name: str = "",
    flagged_label_ids: list = None,
    flagged_centroids: dict = None,
    on_mark_tier=None,
    on_label_deleted=None,
    flagged_label_items: list = None,
    on_discard_garbage=None,
    on_interpolation_accepted=None,
    on_label_split=None,
    on_label_smoothed=None,
    on_label_merged=None,
    on_recalculate=None,
    on_new_label_painted=None,
    on_batch_refined=None,
    on_batch_refine_lock=None,
    on_add_mode_lock=None,
    on_add_committed=None,
    on_add_undo_session=None,
    on_save_batch_recipe=None,
    get_saved_batch_recipes=None,
    # Reshape R1.  This launcher enumerates every triage
    # callback by name rather than forwarding **kwargs, so a new one
    # cannot reach the widget without three lines here.  De-novo Add
    # needed the same three lines for the same reason.
    on_reshape_lock=None,
    on_refine_lock=None,
    on_reshape_committed=None,
    on_reshape_context=None,
    # Refine recipes: three more names
    # for the same reason as the Reshape ones above.
    on_save_refine_recipe=None,
    get_saved_refine_recipes=None,
    on_delete_refine_recipe=None,
    # Batch Refine "Plot selection (N)" target:
    # a live getter of the ids the QC plot highlights right now.
    get_plot_selection=None,
    # the highest df id over ALL rows (ghosts
    # included), so new Add ids skip the ids of deleted objects.
    get_label_id_floor=None,
    patch_ids=None,
    active_patch_id="",
    switch_patch_callback=None,
    patch_statuses=None,
    prob_volume_path: str = "",
) -> Tuple[bool, str]:
    """Open the 3D curation annotator dock (Curation3DAnnotator subclass).

    Uses a separate dock name from the HITL 2D annotator to prevent collision.
    """
    # Diagnostics: enable persistent crash + breadcrumb logging so a hard
    # Qt/vispy access violation in the 3D annotator (e.g. right-click-nav
    # after a couple of patch switches) leaves a native stack + a breadcrumb
    # trail on disk (~/.epicycle/crash_logs).  The console buffer is lost on a
    # hard crash, so the existing print() diagnostics don't survive — this
    # makes them durable.  Idempotent + best-effort (never blocks launch).
    try:
        from napari_epicycle.utils.crash_logger import (
            install_crash_logging,
            breadcrumb,
        )
        _crash_log_path = install_crash_logging()
        breadcrumb(f"launch_3d_curation_annotator: dock={dock_name!r}")
        if _crash_log_path is not None:
            print(f"[CRASH-LOG] session log -> {_crash_log_path}", flush=True)
    except Exception:
        pass

    if viewer is None:
        return False, "Viewer is not available."

    window = getattr(viewer, "window", None)
    if window is None:
        return False, "Viewer window is not available."

    # 2026-09-04: annotators need synchronous slicing (hide -> recolour ->
    # re-show within one tick).  Turn napari's persisted experimental.async_
    # OFF unless a Step-1 progressive loader is live on this viewer.
    try:
        from packages.epicycle_largedata.progressive import ensure_sync_slicing
        ensure_sync_slicing(viewer, reason="3D curation annotator launch")
    except Exception:
        pass

    def _wire_patch_options(widget):
        if hasattr(widget, "set_patch_options"):
            widget.set_patch_options(
                patch_ids=patch_ids or [],
                active_patch_id=active_patch_id,
                switch_patch_callback=switch_patch_callback,
                patch_statuses=patch_statuses,
            )

    def _wire_widget_runtime(target_widget):
        _wire_patch_options(target_widget)
        if hasattr(target_widget, "set_context"):
            target_widget.set_context(
                image_layer_name=image_layer_name,
                labels_layer_name=labels_layer_name,
                auto_load=True,
            )
        if flagged_label_ids and hasattr(target_widget, "set_focused_refinement"):
            target_widget.set_focused_refinement(
                flagged_label_ids or [], flagged_centroids or {},
            )
        if hasattr(target_widget, "set_triage_callbacks"):
            target_widget.set_triage_callbacks(
                on_mark_tier=on_mark_tier,
                on_label_deleted=on_label_deleted,
                on_merge=on_label_merged,
            )
        if flagged_label_items and hasattr(target_widget, "set_flagged_labels"):
            target_widget.set_flagged_labels(flagged_label_items)
        if on_discard_garbage and hasattr(target_widget, "set_discard_garbage_callback"):
            target_widget.set_discard_garbage_callback(on_discard_garbage)
        if hasattr(target_widget, "set_curation_callbacks"):
            target_widget.set_curation_callbacks(
                on_interpolation_accepted=on_interpolation_accepted,
                on_label_split=on_label_split,
                on_label_smoothed=on_label_smoothed,
                on_recalculate=on_recalculate,
                on_new_label_painted=on_new_label_painted,
                on_batch_refined=on_batch_refined,
                on_batch_refine_lock=on_batch_refine_lock,
                on_add_mode_lock=on_add_mode_lock,
                on_add_committed=on_add_committed,
                on_add_undo_session=on_add_undo_session,
                on_save_batch_recipe=on_save_batch_recipe,
                get_saved_batch_recipes=get_saved_batch_recipes,
                on_reshape_lock=on_reshape_lock,
                on_refine_lock=on_refine_lock,
                on_reshape_committed=on_reshape_committed,
                on_reshape_context=on_reshape_context,
                on_save_refine_recipe=on_save_refine_recipe,
                get_saved_refine_recipes=get_saved_refine_recipes,
                on_delete_refine_recipe=on_delete_refine_recipe,
                get_plot_selection=get_plot_selection,
                get_label_id_floor=get_label_id_floor,
            )
        if prob_volume_path and hasattr(target_widget, "set_prob_volume_path"):
            target_widget.set_prob_volume_path(prob_volume_path)

    # Sweep zombie annotator widgets from previous close/relaunch cycles
    # that escaped the close hook's deleteLater() call.
    _sweep_orphan_annotator_widgets()

    # Resolve all docks (not just one). Hidden duplicates can keep stale
    # annotator objects alive and cause progressive lag across relaunches.
    _purge_empty_docks(viewer, dock_name)
    docks = _find_all_docks_by_title(viewer, dock_name)
    if not docks:
        dock = _resolve_existing_dock(viewer, dock_name)
        if dock is not None:
            docks = [dock]

    visible_dock = None
    hidden_docks = []
    for dock in docks:
        try:
            is_visible = bool(dock.isVisible())
        except Exception:
            is_visible = False
        if is_visible and visible_dock is None:
            visible_dock = dock
        else:
            hidden_docks.append(dock)

    # Always purge hidden duplicates first.
    for dock in hidden_docks:
        try:
            inner = dock.widget() if hasattr(dock, "widget") else dock
            _cleanup_old_annotator_widget(inner)
        except Exception:
            pass
        _destroy_dock(viewer, dock)

    if visible_dock is not None:
        try:
            target_widget = (
                visible_dock.widget()
                if hasattr(visible_dock, "widget")
                else visible_dock
            )
            _install_dock_close_hook(visible_dock, target_widget)
            _wire_widget_runtime(target_widget)
            if hasattr(visible_dock, "raise_"):
                visible_dock.raise_()
        except Exception:
            pass
        return True, "3D curation annotator is already open."

    # No visible annotator dock remains: destroy all satellite docks
    # (navigator, advanced features, multiviewer settings) to prevent
    # accumulation across close/relaunch cycles.
    _destroy_satellite_docks(viewer)
    _close_orphan_magicgui_windows_global()

    try:
        module = importlib.import_module(
            "napari_epicycle.annotator.curation_3d_widget"
        )
        widget_cls = getattr(module, "Curation3DAnnotator")
        widget = widget_cls(viewer)
        _wire_widget_runtime(widget)
        window.add_dock_widget(widget, name=dock_name, area="right")
        dock = (
            _find_dock_by_title(viewer, dock_name)
            or _resolve_existing_dock(viewer, dock_name)
        )
        _tabify_native_with_epicycle(viewer, dock, dock_name, retries=6)
        if dock is not None:
            _install_dock_close_hook(dock, widget)
        return True, "Opened 3D curation annotator."
    except Exception as exc:
        return False, f"Failed to open 3D curation annotator: {exc}"

def _setup_standalone_auto_load(viewer, widget) -> None:
    """Wire deferred auto-load and layer-insertion listener for standalone mode."""
    from qtpy.QtCore import QTimer

    def _try_auto_load():
        try:
            widget._refresh_layer_options()
            img = widget.image_combo.currentText()
            lbl = widget.labels_combo.currentText()
            if img and not lbl:
                # Auto-create a labels layer matching the first image.
                widget._create_labels_from_image()
                lbl = widget.labels_combo.currentText()
            if img and lbl:
                widget._load_selected_layers()
            elif img:
                widget.status_label.setText(
                    "Image found but labels creation failed — add a labels layer."
                )
            else:
                widget.status_label.setText(
                    "No layers found — drag images into the viewer."
                )
        except Exception:
            pass

    def _on_layer_inserted(_event=None):
        if getattr(widget, "_active_labels_name", ""):
            return
        QTimer.singleShot(200, _try_auto_load)

    QTimer.singleShot(150, _try_auto_load)
    viewer.layers.events.inserted.connect(_on_layer_inserted)
    # Store reference on the widget so _disconnect_viewer_events can
    # disconnect it on close, preventing handler accumulation.
    widget._standalone_layer_inserted_cb = _on_layer_inserted


class _StandaloneInferenceDialog:
    """Modal dialog for setting cellpose inference parameters."""

    @staticmethod
    def get_parameters(parent, current_config: dict) -> dict | None:
        """Show a dialog and return updated config, or None if cancelled."""
        from qtpy.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QSpinBox,
            QCheckBox,
            QComboBox,
            QVBoxLayout,
            QLabel,
        )

        dlg = QDialog(parent)
        dlg.setWindowTitle("Inference Parameters")
        dlg.setMinimumWidth(320)
        layout = QVBoxLayout(dlg)

        header = QLabel("<b>Cellpose Inference Parameters</b>")
        layout.addWidget(header)

        form = QFormLayout()

        flow_sb = QDoubleSpinBox()
        flow_sb.setRange(0.0, 10.0)
        flow_sb.setSingleStep(0.1)
        flow_sb.setDecimals(2)
        flow_sb.setValue(float(current_config.get("flow_threshold", 0.4)))
        form.addRow("Flow threshold:", flow_sb)

        cellprob_sb = QDoubleSpinBox()
        cellprob_sb.setRange(-10.0, 10.0)
        cellprob_sb.setSingleStep(0.5)
        cellprob_sb.setDecimals(2)
        cellprob_sb.setValue(float(current_config.get("cellprob_threshold", 0.0)))
        form.addRow("Cell prob threshold:", cellprob_sb)

        diameter_sb = QSpinBox()
        diameter_sb.setRange(0, 500)
        diameter_sb.setValue(int(current_config.get("diameter", 0) or 0))
        diameter_sb.setSpecialValueText("auto")
        form.addRow("Diameter (0 = auto):", diameter_sb)

        ch1_combo = QComboBox()
        ch1_combo.addItems(["0: grayscale", "1: Red", "2: Green", "3: Blue"])
        ch1_val = str(current_config.get("channel1", "2: Green"))
        idx = ch1_combo.findText(ch1_val)
        if idx >= 0:
            ch1_combo.setCurrentIndex(idx)
        form.addRow("Channel 1 (cyto):", ch1_combo)

        ch2_combo = QComboBox()
        ch2_combo.addItems(["0: None", "1: Red", "2: Green", "3: Blue"])
        ch2_val = str(current_config.get("channel2", "0: None"))
        idx = ch2_combo.findText(ch2_val)
        if idx >= 0:
            ch2_combo.setCurrentIndex(idx)
        form.addRow("Channel 2 (nuclei):", ch2_combo)

        normalize_cb = QCheckBox()
        normalize_cb.setChecked(bool(current_config.get("normalize", True)))
        form.addRow("Normalize:", normalize_cb)

        rescale_cb = QCheckBox()
        rescale_cb.setChecked(bool(current_config.get("rescale", True)))
        form.addRow("Rescale:", rescale_cb)

        invert_cb = QCheckBox()
        invert_cb.setChecked(bool(current_config.get("invert", False)))
        form.addRow("Invert:", invert_cb)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec_() != QDialog.Accepted:
            return None

        return {
            "flow_threshold": flow_sb.value(),
            "cellprob_threshold": cellprob_sb.value(),
            "diameter": diameter_sb.value(),
            "channel1": ch1_combo.currentText(),
            "channel2": ch2_combo.currentText(),
            "normalize": normalize_cb.isChecked(),
            "rescale": rescale_cb.isChecked(),
            "invert": invert_cb.isChecked(),
        }


class _StandaloneSegmentHelper:
    """Lightweight cellpose segmentation for the standalone 2D annotator.

    On the first segment request, a file dialog prompts the user to choose a
    cellpose model.  Subsequent calls reuse the same model.  The helper
    extracts the 2D image slice for the requested orientation/index from the
    viewer's Image layers, runs inference, and writes the mask back into the
    Labels layer.
    """

    def __init__(self, viewer, widget) -> None:
        self._viewer = viewer
        self._widget = widget
        self._segmenter = None
        self._model_path: str = ""
        self._config: dict = {
            "model_type": "custom",
            "pretrained_model_path": "",
            "device": "GPU",
            "channel1": "2: green",
            "channel2": "1: red",
            "normalize": True,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "rescale": True,
            "diameter": 0,
            "invert": False,
        }

    def _ensure_model(self) -> bool:
        """Lazily load a cellpose model, prompting for path and params."""
        from qtpy.QtWidgets import QFileDialog

        if self._segmenter is not None:
            return True

        path, _ = QFileDialog.getOpenFileName(
            self._widget, "Select Cellpose Model", "", "All Files (*)"
        )
        if not path:
            return False

        # Show inference parameter dialog so user can set channels, thresholds, etc.
        params = _StandaloneInferenceDialog.get_parameters(
            self._widget, self._config
        )
        if params is None:
            return False
        self._config.update(params)

        self._model_path = path
        self._config["pretrained_model_path"] = path
        try:
            from napari_epicycle.segmenters.cellpose_segmenter import (
                CellposeSegmenter,
            )

            self._segmenter = CellposeSegmenter(self._config)
        except Exception as exc:
            self._widget.status_label.setText(f"Model load failed: {exc}")
            return False
        return True

    def _get_layer(self, name: str):
        """Get a layer by name from the viewer, or None."""
        try:
            return self._viewer.layers[name]
        except (KeyError, IndexError):
            return None

    def _extract_2d_slice(self, orientation: str, slice_index: int):
        """Return a 2D (or multi-channel 2D) image array for the slice.

        If multiple Image layers are present, stacks them as channels.
        """
        from napari.layers import Image

        image_layers = []
        for layer in self._viewer.layers:
            if isinstance(layer, Image) and layer.data.ndim in {2, 3}:
                image_layers.append(layer)

        if not image_layers:
            return None

        slices = []
        for layer in image_layers:
            data = np.asarray(layer.data)
            if data.ndim == 2:
                # Single 2D image — use as-is regardless of orientation.
                slices.append(data)
                continue
            # 3D stack: (Z, H, W)
            z_max, h_max, w_max = data.shape
            if orientation == "XY":
                idx = min(max(0, slice_index), z_max - 1)
                slices.append(data[idx])
            elif orientation == "XZ":
                idx = min(max(0, slice_index), h_max - 1)
                slices.append(data[:, idx, :])
            elif orientation == "YZ":
                idx = min(max(0, slice_index), w_max - 1)
                slices.append(data[:, :, idx])

        if not slices:
            return None
        if len(slices) == 1:
            return slices[0]
        # Stack multiple layers → (H, W, C)
        return np.stack(slices, axis=-1)

    def __call__(self, orientation: str, slice_index: int | None = None) -> dict:
        orient = str(orientation or "").upper()
        if orient not in {"XY", "XZ", "YZ"}:
            return {"ok": False, "message": f"Unsupported orientation: {orientation}"}

        if not self._ensure_model():
            return {"ok": False, "message": "No model selected."}

        # Resolve current slice from the widget's subviewer if not provided.
        if slice_index is None:
            vm = self._widget._viewer_model_for_orientation(orient)
            slice_index = self._widget._safe_first_dim_index(vm) if vm else 0

        img_name = self._widget.image_combo.currentText()
        lbl_name = self._widget.labels_combo.currentText()
        lbl_layer = self._get_layer(lbl_name)
        if lbl_layer is None:
            return {"ok": False, "message": "Labels layer not found in viewer."}

        img_slice = self._extract_2d_slice(orient, slice_index)
        if img_slice is None:
            return {"ok": False, "message": "Could not extract image slice."}

        try:
            masks, _flows, _probs = self._segmenter.segment(img_slice, self._config)
        except Exception as exc:
            return {"ok": False, "message": f"Segmentation failed: {exc}"}

        # Write mask back into the labels layer.
        lbl_data = np.asarray(lbl_layer.data)
        if lbl_data.ndim == 3:
            z_max, h_max, w_max = lbl_data.shape
            if orient == "XY":
                idx = min(max(0, slice_index), z_max - 1)
                lbl_data[idx] = masks
            elif orient == "XZ":
                idx = min(max(0, slice_index), h_max - 1)
                lbl_data[:, idx, :] = masks
            elif orient == "YZ":
                idx = min(max(0, slice_index), w_max - 1)
                lbl_data[:, :, idx] = masks
        elif lbl_data.ndim == 2:
            lbl_data[:] = masks
        lbl_layer.data = lbl_data
        lbl_layer.refresh()

        return {
            "ok": True,
            "message": f"Segmented {orient} slice {slice_index} ({int(masks.max())} labels).",
            "slice_index": slice_index,
            "image_layer_name": img_name,
            "labels_layer_name": lbl_name,
        }


def launch_standalone_2d_annotator(
    viewer,
    *,
    dock_name: str = "EpiCycle 2D Annotator",
    image_layer_name: str = "",
    labels_layer_name: str = "",
) -> Tuple[bool, str]:
    """Open the 2D multiview annotator in standalone mode (no HITL pipeline).

    Auto-detects Image/Labels layers in the viewer.  A lightweight cellpose
    segmentation callback is wired automatically — clicking Segment for the
    first time will prompt for a model file via a file dialog.  Accept and
    Overlay buttons are non-functional (no HITL pipeline backing them).
    """
    if viewer is None:
        return False, "Viewer is not available."

    window = getattr(viewer, "window", None)
    if window is None:
        return False, "Viewer window is not available."

    _purge_empty_docks(viewer, dock_name)
    dock = _resolve_existing_dock(viewer, dock_name)
    if dock is not None:
        visible = False
        try:
            visible = dock.isVisible()
        except Exception:
            pass
        if visible:
            try:
                target_widget = dock.widget() if hasattr(dock, "widget") else dock
                if hasattr(target_widget, "set_context"):
                    target_widget.set_context(
                        image_layer_name=image_layer_name,
                        labels_layer_name=labels_layer_name,
                        auto_load=True,
                    )
                if hasattr(dock, "raise_"):
                    dock.raise_()
            except Exception:
                pass
            return True, "2D annotator is already open."
        else:
            try:
                inner = dock.widget() if hasattr(dock, "widget") else dock
                _cleanup_old_annotator_widget(inner)
            except Exception:
                pass
            _destroy_dock(viewer, dock)

    try:
        module = importlib.import_module(
            "napari_epicycle.annotator.native_multiview_widget"
        )
        widget_cls = getattr(module, "NativeMultiviewAnnotator")
        widget = widget_cls(viewer)

        # Wire standalone segmentation: on first Segment click, a file
        # dialog prompts for a cellpose model.
        seg_helper = _StandaloneSegmentHelper(viewer, widget)
        if hasattr(widget, "set_runtime_callbacks"):
            widget.set_runtime_callbacks(segment_slice_callback=seg_helper)

        if image_layer_name or labels_layer_name:
            if hasattr(widget, "set_context"):
                widget.set_context(
                    image_layer_name=image_layer_name,
                    labels_layer_name=labels_layer_name,
                    auto_load=True,
                )
        window.add_dock_widget(widget, name=dock_name, area="right")
        dock = (
            _find_dock_by_title(viewer, dock_name)
            or _resolve_existing_dock(viewer, dock_name)
        )
        _tabify_native_with_epicycle(viewer, dock, dock_name, retries=6)

        # Deferred auto-load: if layers already present, load them into
        # subviewers after the dock is attached.  Also listen for new
        # layers being added so drag-and-drop works.
        _setup_standalone_auto_load(viewer, widget)

        return True, "Opened 2D annotator."
    except Exception as exc:
        return False, f"Failed to open 2D annotator: {exc}"


_ADVANCED_DOCK_NAMES = {
    "hitl_2d": "EpiCycle 2D Advanced Features",
    "3d_curation": "EpiCycle 3D Advanced Features",
}


def launch_advanced_features_dock(
    viewer,
    *,
    dock_name: str | None = None,
    mode: str = "hitl_2d",
    split_callbacks: dict | None = None,
    smooth_callbacks: dict | None = None,
) -> Tuple:
    """Open the Advanced Features dock, reusing an existing one if present.

    Parameters
    ----------
    mode : str
        ``"hitl_2d"`` for HITL 2D (SAM, Otsu, Watershed, multiviewer).
        ``"3d_curation"`` for full napari-multiview-annotator workflows.
    split_callbacks : dict or None
        Keyword arguments for ``widget.set_split_callbacks()``.  Passed
        when ``mode="3d_curation"`` to wire the Assisted Split UI to
        the annotator's orchestration methods.
    smooth_callbacks : dict or None
        Keyword arguments for ``widget.set_smooth_callbacks()``.  Passed
        when ``mode="3d_curation"`` to wire the Smooth Label UI.

    Returns ``(widget_or_None, success_bool, message_str)``.
    """
    if dock_name is None:
        dock_name = _ADVANCED_DOCK_NAMES.get(mode, "EpiCycle Advanced Features")

    if viewer is None:
        return None, False, "Viewer is not available."

    window = getattr(viewer, "window", None)
    if window is None:
        return None, False, "Viewer window is not available."

    # A dock left empty by napari's X (or holding something that is not the
    # panel) is removed here and a fresh panel is built below, as on a first
    # launch (live driver, part G / F5).
    def _is_panel(inner):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )
        return isinstance(inner, AdvancedFeaturesWidget)

    _purge_empty_docks(viewer, dock_name, accept=_is_panel)

    dock = _resolve_existing_dock(viewer, dock_name)
    if dock is not None:
        try:
            widget = dock.widget() if hasattr(dock, "widget") else dock
            if split_callbacks and hasattr(widget, "set_split_callbacks"):
                widget.set_split_callbacks(**split_callbacks)
            if smooth_callbacks and hasattr(widget, "set_smooth_callbacks"):
                widget.set_smooth_callbacks(**smooth_callbacks)
            if hasattr(dock, "show"):
                dock.show()
            if hasattr(dock, "raise_"):
                dock.raise_()
            return widget, True, "Advanced Features dock is already open."
        except Exception:
            return None, False, "Failed to reuse existing Advanced Features dock."

    try:
        module = importlib.import_module(
            "napari_epicycle.annotator.advanced_features_widget"
        )
        widget_cls = getattr(module, "AdvancedFeaturesWidget")
        widget = widget_cls(mode=mode)
        if split_callbacks and hasattr(widget, "set_split_callbacks"):
            widget.set_split_callbacks(**split_callbacks)
        if smooth_callbacks and hasattr(widget, "set_smooth_callbacks"):
            widget.set_smooth_callbacks(**smooth_callbacks)
        window.add_dock_widget(widget, name=dock_name, area="right")
        dock = (
            _find_dock_by_title(viewer, dock_name)
            or _resolve_existing_dock(viewer, dock_name)
        )
        if dock is not None:
            # Launch as a floating window so it doesn't crowd the main layout.
            try:
                dock.setFloating(True)
                dock.resize(340, 350)
            except Exception:
                pass
            if hasattr(dock, "show"):
                dock.show()
            if hasattr(dock, "raise_"):
                dock.raise_()
        return widget, True, "Opened Advanced Features dock."
    except Exception as exc:
        return None, False, f"Failed to open Advanced Features dock: {exc}"


# -----------------------------------------------------------------------
# Multiviewer Settings — floating display-settings dock
# -----------------------------------------------------------------------

_MULTIVIEWER_SETTINGS_DOCK_NAME = "EpiCycle Multiviewer Settings"


def launch_multiviewer_settings_dialog(viewer) -> Tuple:
    """Open or raise the Multiviewer Settings dock.

    Returns ``(widget_or_None, success_bool, message_str)``.
    """
    dock_name = _MULTIVIEWER_SETTINGS_DOCK_NAME

    if viewer is None:
        return None, False, "Viewer is not available."

    window = getattr(viewer, "window", None)
    if window is None:
        return None, False, "Viewer window is not available."

    _purge_empty_docks(viewer, dock_name)

    dock = _resolve_existing_dock(viewer, dock_name)
    if dock is not None:
        try:
            if hasattr(dock, "show"):
                dock.show()
            if hasattr(dock, "raise_"):
                dock.raise_()
            widget = dock.widget() if hasattr(dock, "widget") else dock
            return widget, True, "Multiviewer Settings already open."
        except Exception:
            return None, False, "Failed to reuse existing Multiviewer Settings."

    try:
        module = importlib.import_module(
            "napari_epicycle.annotator.multiviewer_settings_dialog"
        )
        widget_cls = getattr(module, "MultiviewerSettingsWidget")
        widget = widget_cls()
        window.add_dock_widget(widget, name=dock_name, area="right")
        dock = (
            _find_dock_by_title(viewer, dock_name)
            or _resolve_existing_dock(viewer, dock_name)
        )
        if dock is not None:
            try:
                dock.setFloating(True)
                dock.resize(280, 180)
            except Exception:
                pass
            if hasattr(dock, "show"):
                dock.show()
            if hasattr(dock, "raise_"):
                dock.raise_()
        return widget, True, "Opened Multiviewer Settings."
    except Exception as exc:
        return None, False, f"Failed to open Multiviewer Settings: {exc}"


# -----------------------------------------------------------------------
# Label Navigator — floating triage command center for 3D curation
# -----------------------------------------------------------------------

_NAVIGATOR_DOCK_NAME = "EpiCycle Label Navigator"


def launch_curation_label_navigator(
    viewer,
    annotator_widget,
    *,
    dock_name: str = _NAVIGATOR_DOCK_NAME,
) -> Tuple:
    """Open the Label Navigator as a floating dock.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    annotator_widget : Curation3DAnnotator
        The annotator that owns this navigator.  Used to create the
        navigator widget with properly-wired callbacks.
    dock_name : str
        Dock title (used for duplicate prevention).

    Returns ``(widget_or_None, success_bool, message_str)``.
    """
    import time as _time

    _t0 = _time.perf_counter()

    if viewer is None:
        return None, False, "Viewer is not available."

    window = getattr(viewer, "window", None)
    if window is None:
        return None, False, "Viewer window is not available."

    _purge_empty_docks(viewer, dock_name)

    dock = _resolve_existing_dock(viewer, dock_name)
    if dock is not None:
        _curation_debug(
            f"navigator reuse path dock_type={type(dock).__name__} "
            f"visible={getattr(dock, 'isVisible', lambda: False)()}"
        )
        try:
            widget = dock.widget() if hasattr(dock, "widget") else dock
            if hasattr(dock, "show"):
                dock.show()
            if hasattr(dock, "raise_"):
                dock.raise_()

            # Rebind navigator callbacks to the current annotator instance.
            # After close/relaunch the navigator dock survives in Qt memory
            # but its callbacks pointed to the old (destroyed) annotator.
            if hasattr(annotator_widget, "_wire_navigator_callbacks"):
                try:
                    annotator_widget._wire_navigator_callbacks(widget)
                except Exception:
                    pass

            # Reset stale UI state (selection + isolate checkbox) from
            # the previous session before pushing new items.
            if hasattr(widget, "reset_selection_ui"):
                try:
                    widget.reset_selection_ui()
                except Exception:
                    pass

            # Update annotator reference and push cached items.
            # Always call set_items — even for empty cache — to clear
            # stale rows from the previous session.
            annotator_widget._navigator_widget = widget
            annotator_widget._navigator_dock = dock
            # Pass the active Labels layer so the navigator's swatch
            # column samples real on-screen colors on first paint
            # instead of falling back to neutral gray.
            try:
                if hasattr(widget, "set_label_layer"):
                    layer = annotator_widget._layer_by_name(
                        annotator_widget.viewer.layers,
                        annotator_widget._active_labels_name,
                    )
                    widget.set_label_layer(layer)
            except Exception:
                pass
            cached = getattr(annotator_widget, "_cached_flagged_items", None)
            try:
                widget.set_items(cached if cached is not None else [])
            except Exception:
                pass

            _dt = _time.perf_counter() - _t0
            if _dt > 0.03:
                _curation_debug(
                    "navigator reuse complete "
                    f"dt={_dt:.3f}s widget_type={type(widget).__name__} "
                    f"id={id(widget)}"
                )
            return widget, True, "Label Navigator is already open."
        except Exception:
            return None, False, "Failed to reuse existing Label Navigator dock."

    try:
        widget = annotator_widget._create_navigator()
        window.add_dock_widget(widget, name=dock_name, area="right")
        dock = (
            _find_dock_by_title(viewer, dock_name)
            or _resolve_existing_dock(viewer, dock_name)
        )
        if dock is not None:
            try:
                dock.setFloating(True)
                dock.resize(520, 480)
            except Exception:
                pass
            if hasattr(dock, "show"):
                dock.show()
            if hasattr(dock, "raise_"):
                dock.raise_()

        # Store reference on annotator and push cached items.
        annotator_widget._navigator_widget = widget
        annotator_widget._navigator_dock = dock
        # Pass the active Labels layer so the navigator's swatch column
        # samples real on-screen colors on first paint instead of
        # falling back to neutral gray.
        try:
            if hasattr(widget, "set_label_layer"):
                layer = annotator_widget._layer_by_name(
                    annotator_widget.viewer.layers,
                    annotator_widget._active_labels_name,
                )
                widget.set_label_layer(layer)
        except Exception:
            pass
        cached = getattr(annotator_widget, "_cached_flagged_items", None)
        if cached:
            widget.set_items(cached)

        _dt = _time.perf_counter() - _t0
        if _dt > 0.03:
            _curation_debug(
                "navigator open new dock complete "
                f"dt={_dt:.3f}s widget_type={type(widget).__name__} "
                f"id={id(widget)}"
            )
        return widget, True, "Opened Label Navigator."
    except Exception as exc:
        return None, False, f"Failed to open Label Navigator: {exc}"

