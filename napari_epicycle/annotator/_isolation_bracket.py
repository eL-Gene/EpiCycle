"""The isolation a finished edit leaves behind, as a pure rule.

No Qt and no napari: the widget records what was isolated before an edit
(``IsolationSnapshot``) and asks ``restore_target`` which cells to isolate
afterwards.  Keeping the rule here lets it be tested on plain values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# How many new pieces are shown together after a cell is split.  Each piece
# costs a small 3D layer (about a quarter of a second), so above the cap only
# the largest piece is shown (the caller lists the biggest first).  The
# Reshape tab exposes this as a setting; this value is its default and the
# fallback when the setting cannot be read.
DEFAULT_PIECES_CAP = 10

MODE_OFF = "off"
MODE_SINGLE = "single"
MODE_MULTI = "multi"


@dataclass(frozen=True)
class IsolationSnapshot:
    """What the isolation looked like when an edit began.

    ``mode`` is ``"off"``, ``"single"`` or ``"multi"``; ``ids`` the isolated
    label ids; ``main_visible`` whether the main labels layer was shown;
    ``checkbox`` the navigator's Isolate box.
    """

    mode: str
    ids: frozenset
    main_visible: bool
    checkbox: bool


@dataclass(frozen=True)
class OpenBracket:
    """An edit's open bracket: which kind of edit owns it, the snapshot
    taken when that edit began and the cell the edit runs on."""

    kind: str
    snapshot: IsolationSnapshot
    target: int = 0


def biggest_first(ids: Iterable[int], voxels: dict) -> tuple:
    """*ids* with at least one voxel, biggest first (ties by id).

    *voxels* maps an id to its voxel count in the written volume.  An id
    missing from it, or with zero voxels, is not a cell and is left out, so it
    cannot use up the pieces cap.
    """
    kept = [int(i) for i in ids if int(voxels.get(int(i), 0)) > 0]
    return tuple(sorted(kept, key=lambda i: (-int(voxels[i]), i)))


def restore_target(
    snapshot: IsolationSnapshot,
    target: int,
    consumed: Iterable[int],
    created: tuple,
    *,
    pieces_cap: int,
) -> frozenset:
    """The ids to isolate after an edit on *target*.

    *consumed* are ids the edit removed (empty when the original survives);
    *created* are the new ids, in the order the caller wants them kept (biggest
    first), so "the first created id" is defined.  *pieces_cap* has no
    default here: the caller supplies it.
    The result can be empty (every isolated cell was consumed and nothing
    replaced it); the caller then isolates nothing.
    """
    if snapshot.mode == MODE_OFF:
        return frozenset()
    ids = frozenset(int(i) for i in snapshot.ids)
    created = tuple(int(c) for c in created)
    kept = ids - frozenset(int(c) for c in consumed)
    if int(target) not in ids or not created:
        return kept
    if len(created) <= pieces_cap:
        return kept | frozenset(created)
    return kept | frozenset({created[0]})
