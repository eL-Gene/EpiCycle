"""The quality gate -- four plain sentences about a finished cut.

The July sandbox showed three warnings after every cut.  The panel showed
none of them, not because the numbers were missing but because they were
thrown away: ``ReshapeResult`` already carries ``coverage``,
``piece_sizes``, ``n_seeds_found`` and ``n_seeds``, and nothing read them.

Four rules, in the order the user reads them:

(a) **a marker got no piece of its own** -- two markers landed in the same
    cell, or one was absorbed.  Suppressed when ``merge_to_count`` fired,
    because that stage absorbs marker-backed pieces *on purpose* and the
    engine already says so; a second warning would be a complaint about a
    thing the user asked for.
(b) **coverage** -- voxels the cut gave to nobody.  They are not lost: they
    stay with the target cell on Accept, and the sentence says so.
(c) **a sliver** -- the ``0.2 x median`` rule, **moved** out of
    ``update_split_guided_ui`` with its arithmetic untouched.  The widget
    still unticks the pieces named here, but it no longer decides WHICH:
    the Reshape tab now has one copy of the rule and it is this one.
    (The Add tab keeps its own sliver handling -- a different question on
    a different list, and out of scope for this brick.)
(d) **the expected count** -- fewer pieces than the user said they expected.
    Nothing is ever forced on this path.

Pure: no Qt, no napari, stdlib only.  The caller formats; this module
decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "QualityNote",
    "quality_warnings",
    "coverage_percent",
    "shortfall_percent",
    "DEFAULT_COVERAGE_FLOOR",
    "DEFAULT_SLIVER_FRAC",
]

#: Below this share of the cell assigned to pieces, rule (b) speaks.
DEFAULT_COVERAGE_FLOOR = 0.98

#: A piece smaller than this fraction of the median piece is a sliver.
#: The widget's number since Brick 1, moved here unchanged.
DEFAULT_SLIVER_FRAC = 0.2

#: The order the notes are rendered in, top to bottom.
_ORDER = ("lost-marker", "coverage", "sliver", "count")


@dataclass(frozen=True)
class QualityNote:
    """One amber line, and the pieces it is about.

    ``piece_ids`` is empty for a note that names no piece (coverage has no
    piece to point at -- that is the whole complaint).  The panel unticks
    exactly the ids a ``"sliver"`` note carries, which is why they are
    data and not something parsed back out of the sentence.
    """

    kind: str
    text: str
    piece_ids: tuple = field(default_factory=tuple)


def coverage_percent(coverage: float) -> int:
    """The share of the cell the pieces got, as a whole percent.

    S1 and the quality block must agree on ONE number, so both
    read it from here: "94 % assigned" and "6 % not given" are this
    function and its complement, never two roundings of the same float.
    """
    try:
        pct = int(round(float(coverage) * 100.0))
    except Exception:
        return 0
    return max(0, min(100, pct))


def shortfall_percent(coverage: float) -> int:
    """The complement of :func:`coverage_percent` -- they sum to 100."""
    return 100 - coverage_percent(coverage)


def _piece_rows(result: Any) -> list:
    """``(position, id, voxels)`` per piece, in the engine's order.

    ``piece_stats`` is the authority (it carries the ids the panel ticks).
    ``piece_sizes`` alone is enough for the arithmetic, so a result that
    has only the sizes still gets rules (b)-(d) and a sliver note with no
    ids to untick.
    """
    stats = list(getattr(result, "piece_stats", None) or ())
    if stats:
        return [
            (i + 1, int(row.get("id", i + 1)), int(row.get("voxels", 0)))
            for i, row in enumerate(stats)
        ]
    sizes = list(getattr(result, "piece_sizes", None) or ())
    return [(i + 1, None, int(v)) for i, v in enumerate(sizes)]


def _sliver_cut(sizes: Sequence[int], frac: float) -> float:
    """``frac x median(sizes)`` -- the widget's arithmetic, verbatim.

    Including its median: the upper of the two middle values on an even
    count (``svals[len(svals) // 2]``), not their mean.  Moving the rule
    means moving it, so the number a user sees does not change under them.
    """
    svals = sorted(v for v in sizes if v)
    median = svals[len(svals) // 2] if svals else 0
    return float(frac) * median if median > 0 else 0.0


def quality_warnings(
    result: Any,
    *,
    target_id: int,
    coverage_floor: float = DEFAULT_COVERAGE_FLOOR,
    sliver_frac: float = DEFAULT_SLIVER_FRAC,
    expected_count: int | None = None,
    merged_to_count: bool = False,
) -> list:
    """The amber lines this result deserves, in reading order.

    Parameters
    ----------
    result
        A ``ReshapeResult``.  A refused or cancelled one gets NO notes: it
        already carries its own red reason, and a second opinion about a
        cut that never happened is noise.
    target_id
        The cell being reshaped -- named in the coverage sentence, because
        "those voxels stay with cell 149" is the reassurance that makes a
        coverage warning readable instead of alarming.
    expected_count
        What the user ticked "I expect exactly n cells" for, or None.
    merged_to_count
        True when the engine's ``merge_to_count`` stage actually ran and
        merged something -- it suppresses rule (a).
    """
    if getattr(result, "status", "ok") != "ok":
        return []

    notes: list = []
    rows = _piece_rows(result)
    sizes = [vox for _pos, _pid, vox in rows]
    n_pieces = int(getattr(result, "n_pieces", 0) or len(rows))

    # -- (a) a marker with no piece ------------------------------------ #
    n_seeds = int(getattr(result, "n_seeds", 0) or 0)
    lost = n_seeds - n_pieces
    if lost > 0 and not merged_to_count:
        notes.append(QualityNote(
            "lost-marker",
            "{} marker{} did not get a piece of its own -- two markers "
            "probably landed in the same object. Raise 'Min marker gap' or "
            "'Marker blur', or untick the spare marker and Preview "
            "again.".format(lost, "" if lost == 1 else "s"),
        ))

    # -- (b) coverage --------------------------------------------------- #
    coverage = float(getattr(result, "coverage", 0.0) or 0.0)
    if coverage < float(coverage_floor):
        notes.append(QualityNote(
            "coverage",
            "{} % of the object was not given to any piece. Those voxels "
            "stay with object {} when you Accept -- nothing is lost. If it "
            "is more than a few percent, try 'Soft cut', or turn 'Snap to "
            "the wall signal' off.".format(
                shortfall_percent(coverage), int(target_id)),
        ))

    # -- (c) slivers ---------------------------------------------------- #
    cut = _sliver_cut(sizes, sliver_frac)
    if cut:
        for pos, pid, vox in rows:
            if vox >= cut:
                continue
            notes.append(QualityNote(
                "sliver",
                "Piece {} is very small ({:,} voxels) -- unticked for you. "
                "Tick it if it really is an object. If one marker keeps getting "
                "starved, raise 'Balance' under Advanced.".format(pos, vox),
                () if pid is None else (int(pid),),
            ))

    # -- (d) the expected count ----------------------------------------- #
    if expected_count and int(expected_count) > n_pieces:
        notes.append(QualityNote(
            "count",
            "You expected {} objects; {} came out. Nothing was forced. Add "
            "another marker, or lower 'Prominence' under Advanced.".format(
                int(expected_count), n_pieces),
        ))

    notes.sort(key=lambda n: _ORDER.index(n.kind) if n.kind in _ORDER else 99)
    return notes
