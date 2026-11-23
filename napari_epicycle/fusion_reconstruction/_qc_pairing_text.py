"""Every sentence of the pairing check that a user can read.

The pairing check compares the active run of a patch with another run of
the same patch during QC feature extraction.  Its words live here, in one
place, so the extraction log, the Extract Features settings, the QC plot
and the help guide say the same thing.  Placeholders are filled with
``str.format``.
"""

from __future__ import annotations

import html

from .qc_pairing import (  # the legend words, defined with the rule
    STATUS_NONE,
    STATUS_NOT_MEASURED,
    STATUS_ONE,
    STATUS_SEVERAL,
    STATUS_SHARED,
    STATUS_TOO_SMALL,
)

#: the row in the Extract Features settings that picks the other run
PICKER_LABEL = "Compare with:"
#: the picker's first item: no comparison
PICKER_NONE = "(none)"
#: the picker's tooltip, on the row's label and on its combo
PICKER_TOOLTIP = (
    "Compare with another run of this patch. "
    "For each object of this run, finds the objects of the other run it "
    "overlaps (for example the nuclei inside each cell) and adds match "
    "columns you can plot, and a Colour by entry named after that run. "
    "Choose (none) to skip.")
#: the dim line under the picker's row, shown with it
PICKER_HINT = (
    "Finds, for each object of this run, the objects of the chosen run it "
    "overlaps, and adds match columns and a Colour by entry named after "
    "that run.")

#: the two run labels files differ in shape
REFUSE_SHAPE = (
    "The match columns were skipped: run '{other}' has labels of {a} "
    "voxels for this patch, and this run has {b}.")
#: the two runs were made from different stack files (file names only)
REFUSE_STACK = (
    "The match columns were skipped: run '{other}' was made from a "
    "different stack file ({other_file}, this run: {this_file}).")
#: the other run's labels file could not be read
REFUSE_UNREADABLE = (
    "The match columns were skipped: the file of run '{other}' could not "
    "be read as labels.")
#: the comparison itself failed (out of memory, a bad file); the reason is
#: the error's own words
REFUSE_FAILED = (
    "The match columns were skipped: run '{other}' could not be compared "
    "({reason}).")
#: the picked run does not list for this patch at dispatch (log only)
NO_LABELS = (
    "Run '{other}' cannot be compared for this patch (no labels, archived "
    "or deleted); match columns skipped.")
#: the log line of a comparison that ran
COMPARED = "Compared with run '{other}': {links} links."

#: the other run's labels file changed after the match values were measured
STALE_MATCH = (
    "Run '{other}' has changed since these match values were measured. "
    "Extract again to refresh them.")
#: the active run was edited after the match values were measured
PREDATE_EDITS = "Match values predate your edits. Extract again to refresh them."


#: the Colour by item of the QC plot and of the triage histogram, naming
#: the other run
COLOUR_BY_MATCH = "Overlap with {other}"
#: the same item when the other run's name is not known
COLOUR_BY_MATCH_UNKNOWN = "Overlap with other run"
#: the stock name of the first run of a workspace ("Overlap with Default"
#: would read as "overlap by default")
DEFAULT_RUN_NAME = "Default"
#: the item when the other run is the stock Default run
COLOUR_BY_MATCH_DEFAULT = "Overlap with the Default run"
#: what every form of the item starts with
_COLOUR_BY_MATCH_PREFIX = "Overlap with "
#: its tooltip
COLOUR_BY_MATCH_TOOLTIP = (
    "Colour by what the other run says about each object: no match, one, "
    "several, or a match shared with another object.")
#: the same item's tooltip in the triage histogram's "Colour bars by"
HIST_COLOUR_BY_MATCH_TOOLTIP = (
    "Stacks the bars by what the other run says about each object: pink "
    "no match, blue one match, gold several matches, cyan shared match; "
    "dark grey too small to judge, light grey not measured.")

#: a match column's tooltip: its name, then its description
COLUMN_TOOLTIP = "{column}: {description}"
#: the words of a match column's description that name the other run
OTHER_RUN = "the other run"
#: the same words with the other run's name
OTHER_RUN_NAMED = "the other run ({other})"
#: the tooltip line naming the other run, for a description that does not
COMPARED_WITH = "Compared with run '{other}'."

#: what the check cannot say (the help guide quotes it)
LIMIT = (
    "One match means the two runs agree, not that the cell is right, and "
    "not that they are the same size: an object nested inside a much larger "
    "one still counts as one match. A merge where both runs merged, or "
    "where the second nucleus mostly lies in the neighbour, is invisible to "
    "this check. Interior membrane brightness is the signal that still "
    "catches it.")
#: the link rule in one sentence (the help guide quotes it)
LINK_RULE = (
    "Two objects are linked when their shared voxels are at least "
    "{link_floor} of the smaller one.")


def link_rule(link_floor: float) -> str:
    """:data:`LINK_RULE` with the floor written as the guide writes it
    (two decimals: ``0.20``)."""
    return LINK_RULE.format(link_floor=f"{float(link_floor):.2f}")


def colour_by_match(other: str = "") -> str:
    """The Colour by item: "Overlap with {other}" with the other run's
    name, "Overlap with the Default run" for the stock Default run,
    "Overlap with other run" when the name is not known."""
    name = str(other or "").strip()
    if not name:
        return COLOUR_BY_MATCH_UNKNOWN
    if name == DEFAULT_RUN_NAME:
        return COLOUR_BY_MATCH_DEFAULT
    return COLOUR_BY_MATCH.format(other=name)


def is_match_colour_by(text) -> bool:
    """Whether ``text`` is the Colour by item of the pairing check, whatever
    other run it names."""
    return isinstance(text, str) and text.startswith(_COLOUR_BY_MATCH_PREFIX)


def column_tooltip(column: str, description: str, other: str = "",
                   notes=()) -> str:
    """A match column's tooltip: ``"{column}: {description}"`` with the
    other run named where the description says "the other run" (or, when
    it does not, a line "Compared with run '{other}'."), then each note
    (the stale line, the predate-edits line) on its own line.

    A run's name is typed by the user, and Qt reads a tooltip that looks
    like markup as rich text: when any line holds ``<``, ``>`` or ``&``,
    the tooltip is written as rich text with every line escaped, so the
    name shows as typed."""
    desc = str(description or "")
    named = False
    if other and OTHER_RUN in desc:
        desc = desc.replace(OTHER_RUN, OTHER_RUN_NAMED.format(other=other), 1)
        named = True
    lines = [COLUMN_TOOLTIP.format(column=column, description=desc)]
    if other and not named:
        lines.append(COMPARED_WITH.format(other=other))
    lines.extend(str(n) for n in notes if n)
    return safe_tooltip(lines)


def safe_tooltip(lines) -> str:
    """*lines* joined as one tooltip: plain text, one line each, or, when
    a line holds ``<``, ``>`` or ``&`` (a run's typed name can), rich text
    with every line escaped and the lines kept apart."""
    lines = [str(line) for line in lines]
    if not any(ch in line for line in lines for ch in "<>&"):
        return "\n".join(lines)
    return "<qt>" + "<br>".join(
        html.escape(line, quote=False) for line in lines) + "</qt>"


def pairing_log_line(outcome) -> "str | None":
    """The extraction log's line for one pairing result: the refusal
    sentence, else "Compared with run '{other}': {links} links.", else
    ``None`` (the comparison did not run)."""
    if not isinstance(outcome, dict) or not outcome:
        return None
    if outcome.get("refused"):
        return str(outcome["refused"])
    other = str(outcome.get("display_name") or outcome.get("run") or "")
    try:
        links = int(outcome.get("links") or 0)
    except (TypeError, ValueError):
        links = 0
    return COMPARED.format(other=other, links=links)


def shape_words(shape) -> str:
    """A volume shape as the refusal says it: ``"40 x 200 x 200"``."""
    return " x ".join(str(int(s)) for s in shape)


__all__ = [
    "PICKER_LABEL", "PICKER_NONE", "PICKER_TOOLTIP", "PICKER_HINT",
    "REFUSE_SHAPE", "REFUSE_STACK", "REFUSE_UNREADABLE", "REFUSE_FAILED",
    "NO_LABELS", "COMPARED", "COMPARED_WITH", "pairing_log_line",
    "STALE_MATCH", "PREDATE_EDITS", "shape_words",
    "STATUS_NONE", "STATUS_ONE", "STATUS_SEVERAL", "STATUS_SHARED",
    "STATUS_TOO_SMALL", "STATUS_NOT_MEASURED",
    "COLOUR_BY_MATCH", "COLOUR_BY_MATCH_UNKNOWN", "colour_by_match",
    "DEFAULT_RUN_NAME", "COLOUR_BY_MATCH_DEFAULT",
    "is_match_colour_by", "COLOUR_BY_MATCH_TOOLTIP",
    "HIST_COLOUR_BY_MATCH_TOOLTIP", "COLUMN_TOOLTIP", "safe_tooltip",
    "OTHER_RUN", "OTHER_RUN_NAMED", "LIMIT", "LINK_RULE", "link_rule",
    "column_tooltip",
]
