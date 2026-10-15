"""Global layer lifecycle manager for the EpiCycle workflow shell.

Centralized, context-aware layer registry that tracks which layers
belong to which workflow step and operation context.  All layer
creation and removal in the main napari viewer should go through
this manager so that:

- Step switching can clean up layers from the previous step
- External layer deletion (trash can) is detected and handled
- Layer ownership is always clear (no orphans, no double-removal)

This is NOT the annotator's internal SubviewerMirrorManager — that
lives in ``packages.epicycle_annotator.subviewer_mirror`` and handles
mirroring main viewer layers to the 3 subviewer panels.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LayerLifecycleManager:
    """Centralized layer registry with context-aware cleanup.

    Contexts are hierarchical strings that map to workflow ownership:

        step1/roi
        step2/patch, step2/predictions, step2/annotator
        step3/view, step3/curation, step3/transient,
        step3/qc/single, step3/qc/population
        step5/segmentation
        step6/tiled

    Parameters
    ----------
    viewer : napari.Viewer
        The main napari viewer instance.
    """

    def __init__(self, viewer) -> None:
        self._viewer = viewer
        # context string → list of layer names
        self._registry: dict[str, list[str]] = {}
        # Callbacks for external removal (user clicked trash can)
        self._removal_callbacks: list[Callable[[str, str], None]] = []
        # Connect to viewer layer removal events
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

    def shutdown(self) -> None:
        """Disconnect from viewer events.  Call on plugin teardown."""
        try:
            self._viewer.layers.events.removed.disconnect(
                self._on_layer_removed
            )
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Creation — thin wrappers that register the layer under a context
    # ------------------------------------------------------------------

    def add_image(self, data, *, name: str, context: str, **kwargs):
        """Add an image layer, registered under the given context."""
        layer = self._viewer.add_image(data, name=name, **kwargs)
        self._register(name, context)
        return layer

    def add_labels(self, data, *, name: str, context: str, **kwargs):
        """Add a labels layer, registered under the given context."""
        layer = self._viewer.add_labels(data, name=name, **kwargs)
        self._register(name, context)
        return layer

    def register_existing(self, layer_name: str, context: str) -> None:
        """Register a layer that was already added to the viewer.

        Use this when layers are created by other code (e.g., annotator
        launcher) but should be tracked by the lifecycle manager.
        """
        if layer_name not in self._viewer.layers:
            return
        self._register(layer_name, context)

    def rename(
        self, old_name: str, new_name: str, new_context: Optional[str] = None
    ) -> None:
        """Update registry after a layer rename, optionally re-contexting."""
        old_ctx = self._find_context(old_name)
        self._unregister(old_name)
        ctx = new_context if new_context is not None else old_ctx
        if ctx:
            self._register(new_name, ctx)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_context(self, context: str) -> None:
        """Remove all layers registered under this context."""
        names = list(self._registry.get(context, []))
        for name in names:
            self._safe_remove(name)
        self._registry.pop(context, None)

    def forget_context(self, context: str) -> None:
        """Remove a context from the registry WITHOUT removing layers.

        Use this during controlled patch switches where another subsystem
        handles the actual layer removal under suppress-flag protection.
        ``clear_context`` would call ``_safe_remove`` (which sets
        ``visible=False``), triggering visibility sync cascades that
        bypass the suppress flags.
        """
        self._registry.pop(context, None)

    def clear_step(self, step_prefix: str) -> None:
        """Remove all layers for a workflow step (e.g., ``'step3'``).

        Matches any context that starts with ``step_prefix``.
        """
        contexts = [c for c in self._registry if c.startswith(step_prefix)]
        for ctx in contexts:
            self.clear_context(ctx)

    def clear_all(self) -> None:
        """Remove all managed layers."""
        for ctx in list(self._registry):
            self.clear_context(ctx)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def layers_in_context(self, context: str) -> list[str]:
        """Return layer names registered under this context."""
        return list(self._registry.get(context, []))

    def context_of(self, layer_name: str) -> Optional[str]:
        """Return the context a layer is registered under, or None."""
        return self._find_context(layer_name)

    def is_managed(self, layer_name: str) -> bool:
        """Return True if this layer is tracked by the manager."""
        return self._find_context(layer_name) is not None

    # ------------------------------------------------------------------
    # External removal handling
    # ------------------------------------------------------------------

    def on_external_removal(
        self, callback: Callable[[str, str], None]
    ) -> None:
        """Register callback(layer_name, context) for trash-can deletions.

        The callback fires when a managed layer is removed by something
        other than ``clear_context`` / ``clear_step`` / ``clear_all``.
        """
        self._removal_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(self, name: str, context: str) -> None:
        names = self._registry.setdefault(context, [])
        if name not in names:
            names.append(name)

    def _unregister(self, name: str) -> None:
        for ctx, names in self._registry.items():
            if name in names:
                names.remove(name)
                return

    def _find_context(self, name: str) -> Optional[str]:
        for ctx, names in self._registry.items():
            if name in names:
                return ctx
        return None

    # Flag to distinguish manager-initiated removals from external ones.
    _removing_internally = False

    def _on_layer_removed(self, event) -> None:
        """Handle layer removal events from the viewer.

        If a managed layer was removed externally (user clicked trash can,
        or another subsystem removed it), fire the removal callbacks so
        controllers can react (e.g., close the annotator gracefully).
        """
        if self._removing_internally:
            return
        layer = getattr(event, "value", None)
        if layer is None:
            return
        name = layer.name if hasattr(layer, "name") else str(layer)
        context = self._find_context(name)
        if context:
            self._unregister(name)
            for cb in self._removal_callbacks:
                try:
                    cb(name, context)
                except Exception:
                    logger.warning(
                        "External removal callback failed for layer %r "
                        "(context=%r)",
                        name,
                        context,
                        exc_info=True,
                    )

    def _safe_remove(self, name: str) -> None:
        """Remove a layer from the viewer with vispy safety pattern."""
        self._removing_internally = True
        try:
            if name in self._viewer.layers:
                layer = self._viewer.layers[name]
                # Hide before remove — disconnects vispy visual from scene
                # graph so no stale GPU texture references remain.
                layer.visible = False
                self._viewer.layers.remove(layer)
        except Exception:
            pass
        finally:
            self._removing_internally = False
        self._unregister(name)
