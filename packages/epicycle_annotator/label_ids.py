"""Helpers for computing fresh label IDs across napari Labels layers.

Background
----------
EpiCycle has multiple places that need to mint a new unused label ID:
  - ``DrawingHandler._on_mouse_drag`` auto-increment after enclosed-fill
  - The Label Navigator's "New Label" button
  - Future: fresh-label helpers in split / merge accept paths

Each of these previously rolled its own ``max(...) + 1`` logic, with
subtle variations.  The bug found 2026-05-16 in the New Label button
(``_next_label_id_after_accept`` returning a navigator-only max which
was LESS than the volume's actual max → collision with existing
labels) showed that "fastest available source of truth" optimizations
silently introduce correctness bugs.

This module centralizes the safe path: scan the actual layer data,
never a derived/filtered view.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def next_unused_label_id(label_layers: Iterable) -> int:
    """Return the next unused positive label ID across all provided
    Labels layers.

    Algorithm: ``max(np.max(layer.data) for each layer) + 1``, with a
    floor of 1 (so we never return 0 = background even on an empty
    volume).

    Parameters
    ----------
    label_layers : iterable
        Any iterable of objects exposing a ``.data`` numpy-like array.
        Empty arrays / arrays without a max are silently skipped.
        Pass the full set of Labels layers in the active viewer to
        guarantee uniqueness across the workspace.

    Notes
    -----
    Scans ALL provided layers — does NOT trust derived state like
    a navigator's ``_items`` list (which may be a tier-filtered view
    that omits some volume labels).  Mirrors the auto-increment logic
    in ``DrawingHandler._on_mouse_drag`` so a New Label button click
    and a post-stroke auto-bump produce IDs from the same arithmetic.

    The function does NOT search for the smallest unused ID — it
    returns ``vol_max + 1`` even if there are gaps (e.g., volume has
    {1, 3, 5} → returns 6, not 2 or 4).  Gap-filling would change
    the semantics from "fresh ID for a new cell" to "reuse a deleted
    label slot" which is a different (and more dangerous) operation.

    Defensive: every layer-data access is wrapped in try/except.  A
    broken layer (no ``.data``, non-numpy array, all-zero on an
    object with no ``size`` attr, etc.) is skipped without error.
    """
    vol_max = 0
    for layer in label_layers:
        try:
            data = getattr(layer, "data", None)
            if data is None:
                continue
            # Avoid np.max on zero-size arrays (raises ValueError).
            size = getattr(data, "size", None)
            if size is None or size == 0:
                continue
            layer_max = int(np.max(data))
            if layer_max > vol_max:
                vol_max = layer_max
        except (AttributeError, TypeError, ValueError):
            continue
    return max(1, vol_max + 1)
