"""Curation selection sync coordinator (U0/U1).

Centralizes selection dispatch across QC plot, flagged list, Label Navigator,
and viewer isolation/navigation.  Introduced in U0 (observability) and U1
(central dispatch) — no user-facing behavior changes.

All surfaces continue to function as before; this module only unifies the
dispatch path and adds optional debug tracing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# U0.1: Canonical SelectionRef
# ---------------------------------------------------------------------------

@dataclass
class SelectionRef:
    """Canonical identity for a curation selection event.

    Every selection action — from plot click, flagged list, navigator,
    or annotator — is normalized into this object before dispatch.
    """

    scope: str                          # "single_patch" or "population"
    patch_key: Optional[str]            # full patch filename key
    label_ids: list[int]                # ordered list (single or multi)
    primary_label_id: Optional[int]     # main label for navigation
    df_indices: list[int]               # DataFrame row indices
    source: str                         # "plot", "flagged_list", "navigator", "annotator"
    mode: str = "focus"                 # "focus" or "isolate"
    timestamp: float = field(default_factory=time.time)

    def _identity_key(self) -> tuple:
        """Key for dedup comparison (excludes timestamp)."""
        return (
            self.scope,
            self.patch_key,
            tuple(self.label_ids),
            self.primary_label_id,
            tuple(self.df_indices),
            self.source,
            self.mode,
        )


# ---------------------------------------------------------------------------
# U1.1: Central dispatch coordinator
# ---------------------------------------------------------------------------

class CurationSelectionSyncCoordinator:
    """Routes selection events to all surfaces with reentrancy + dedup guards.

    Adapters are thin callbacks injected by the controller.  Each adapter
    receives the ``SelectionRef`` and applies the appropriate side effect
    on its surface.

    Source-echo suppression: the surface that originated the selection is
    NOT notified back (prevents feedback loops).
    """

    def __init__(
        self,
        *,
        apply_viewer_isolation: Optional[Callable] = None,
        apply_viewer_navigation: Optional[Callable] = None,
        apply_plot_highlight: Optional[Callable] = None,
        apply_flagged_list_selection: Optional[Callable] = None,
        apply_navigator_selection: Optional[Callable] = None,
    ):
        self._apply_viewer_isolation = apply_viewer_isolation
        self._apply_viewer_navigation = apply_viewer_navigation
        self._apply_plot_highlight = apply_plot_highlight
        self._apply_flagged_list_selection = apply_flagged_list_selection
        self._apply_navigator_selection = apply_navigator_selection

        self._dispatching = False       # reentrancy guard
        self._last_key: Optional[tuple] = None  # dedup
        self._trace_hook: Optional[Callable] = None
        self._rejection_callback: Optional[Callable] = None

    # -- Public API ---------------------------------------------------------

    def set_trace_hook(self, hook: Optional[Callable]) -> None:
        """Set a debug trace callback: ``hook(event_name, sel, extra)``."""
        self._trace_hook = hook

    def update_navigator_adapter(
        self, adapter: Optional[Callable]
    ) -> None:
        """Update the navigator adapter (called when annotator opens/closes)."""
        self._apply_navigator_selection = adapter

    def set_rejection_callback(
        self, callback: Optional[Callable]
    ) -> None:
        """Set callback for rejected selections: ``callback(sel, reason)``."""
        self._rejection_callback = callback

    def dispatch(self, sel: SelectionRef) -> None:
        """Route a selection event to all non-source surfaces.

        Guards:
        1. Reentrancy — nested dispatch calls are silently dropped.
        2. Dedup — identical consecutive dispatches are no-ops.
        3. Population policy — single-label population clicks without
           ``df_indices`` are rejected (existing controller policy).
        """
        # Reentrancy guard
        if self._dispatching:
            return

        # Dedup: identical selection → no-op
        key = sel._identity_key()
        if key == self._last_key:
            return
        self._last_key = key

        # Population policy: reject ambiguous single-label without df_indices
        if (
            sel.scope == "population"
            and len(sel.label_ids) == 1
            and not sel.df_indices
        ):
            self._emit_trace("rejected_population_ambiguous", sel)
            reason = (
                "Ambiguous population click: label may exist in multiple "
                "patches.  Use lasso or select from the flagged list."
            )
            if self._rejection_callback is not None:
                try:
                    self._rejection_callback(sel, reason)
                except Exception:
                    pass
            return

        self._dispatching = True
        try:
            import time as _time

            _t_total = _time.perf_counter()
            self._emit_trace("dispatch", sel)

            is_clear = not sel.label_ids

            # 1. Viewer isolation (always, unless source would conflict)
            _t0 = _time.perf_counter()
            if self._apply_viewer_isolation:
                self._apply_viewer_isolation(sel)
            _t_isolation = _time.perf_counter() - _t0

            # 2. Viewer navigation (skip on clear)
            _t0 = _time.perf_counter()
            if not is_clear and self._apply_viewer_navigation:
                self._apply_viewer_navigation(sel)
            _t_navigation = _time.perf_counter() - _t0

            # 3. Plot highlight — skip if source is plot (avoid echo)
            if sel.source != "plot" and self._apply_plot_highlight:
                self._apply_plot_highlight(sel)

            # 4. Flagged list selection — skip if source is flagged_list
            if sel.source != "flagged_list" and self._apply_flagged_list_selection:
                self._apply_flagged_list_selection(sel)

            # 5. Navigator selection — skip if source is navigator or absent
            if (
                sel.source != "navigator"
                and self._apply_navigator_selection is not None
            ):
                self._apply_navigator_selection(sel)

            _dt_total = _time.perf_counter() - _t_total
            if _dt_total > 0.03:
                print(
                    f"[ISO-TRACE] coordinator dispatch "
                    f"source={sel.source} mode={sel.mode} "
                    f"labels={len(sel.label_ids)} "
                    f"isolation={_t_isolation:.3f}s "
                    f"navigation={_t_navigation:.3f}s "
                    f"total={_dt_total:.3f}s"
                )

        finally:
            self._dispatching = False

    # -- Internal -----------------------------------------------------------

    def _emit_trace(self, event_name: str, sel: SelectionRef,
                    extra: Optional[dict] = None) -> None:
        if self._trace_hook is not None:
            try:
                self._trace_hook(event_name, sel, extra)
            except Exception:
                pass
