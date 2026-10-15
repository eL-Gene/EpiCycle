"""Colormap ownership protocol for 3D curation.

Single-owner authority that prevents controller and annotator from
fighting over ``layer.colormap``.  Any writer must ``claim()`` before
writing and ``release()`` when done.  Ownership transfers are explicit.

See.md for full design.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ColormapOwner(str, Enum):
    DEFAULT = "default"
    CONTROLLER_QC = "controller_qc"
    CONTROLLER_ISOLATION = "controller_isolation"
    ANNOTATOR_ISOLATION = "annotator_isolation"


class ColormapAuthority:
    """Single-owner colormap authority.

    Rules
    -----
    1. ``claim(owner)`` succeeds if no owner or same owner; fails otherwise.
    2. ``release(owner)`` clears ownership only if caller is current owner.
    3. ``set_restore_point`` / ``restore`` manage a single saved colormap.
    4. ``reset()`` clears all state (used on patch switch).
    """

    def __init__(self) -> None:
        self._owner: Optional[ColormapOwner] = None
        self._restore_point: Any = None

    @property
    def current_owner(self) -> Optional[ColormapOwner]:
        return self._owner

    def claim(self, owner: ColormapOwner) -> bool:
        """Claim colormap ownership.  Returns True on success.

        Controller owners (CONTROLLER_QC, CONTROLLER_ISOLATION) can
        freely switch between each other — they are the same system.
        Only annotator ownership blocks controller claims.
        """
        if self._owner is None or self._owner == owner:
            self._owner = owner
            return True
        # Allow controller-to-controller upgrades (same system).
        if self._is_controller(self._owner) and self._is_controller(owner):
            self._owner = owner
            return True
        return False

    @staticmethod
    def _is_controller(owner: ColormapOwner) -> bool:
        return owner in (ColormapOwner.CONTROLLER_QC, ColormapOwner.CONTROLLER_ISOLATION)

    def release(self, owner: ColormapOwner) -> None:
        """Release ownership (no-op if caller is not current owner)."""
        if self._owner == owner:
            self._owner = None

    def set_restore_point(self, colormap: Any) -> None:
        """Save a colormap to restore later."""
        self._restore_point = colormap

    def restore(self) -> Any:
        """Return and clear the saved restore point."""
        cmap = self._restore_point
        self._restore_point = None
        return cmap

    def reset(self) -> None:
        """Clear all state (owner + restore point).  Used on patch switch."""
        self._owner = None
        self._restore_point = None
