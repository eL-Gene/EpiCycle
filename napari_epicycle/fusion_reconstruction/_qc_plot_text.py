"""Every word of Step 3's QC plot area.

Qt-free on purpose: a test reads it without a QApplication, and the plot
module (``qc_plot_widget.py``) never imports it -- the host hands
:data:`STEP3_PLOT_TEXTS` in through ``QCPlotWidget(texts=...)``, the way
Step 4 hands in ``WORKBENCH_PLOT_TEXTS``.

One noun inside the area: **labels** (never "cells"); one spelling:
colours.  A sentence with a title and a body is written ``"title|body"``
(the plot splits on ``"|"``).  Each constant is used as it is or with
``.format(...)``; a count has a singular twin ending ``_ONE``.
"""

from __future__ import annotations

# --- The plot's own words (``QCPlotWidget(layout="workbench", texts=)``) ---

#: The plot's eight help lines, under the keys the Workbench layout reads
#: (``qc_plot_widget.WB_HELP_KEYS``).  "Removed" is the key of the ghosts
#: toggle, whose word in Step 3 is :data:`GHOSTS_WORD`.
PLOT_HELP_LINES = {
    "Plot type": "Choose how to lay out the labels: two features, the PCA "
                 "map, the UMAP or t-SNE map, or a violin per group.",
    "Lasso": "Drag a loop around labels to select them. Click without "
             "dragging to select one label.",
    "Density": "Draw contour lines where labels are packed densely.",
    "Removed": "Show the ghosts: labels deleted or replaced in earlier "
               "edits, drawn as small grey crosses behind the others. They "
               "cannot be selected.",
    "Clear": "Deselect all labels. Their tiers are not changed.",
    "Colour by": "Colour the points by a tier, a cluster, border labels or "
                 "any measured feature.",
    "Scale": "How feature values map to colours: Linear, Log, or Rank "
             "(an even spread, the default).",
    "Colours": "The colour scale used for a feature.",
}

#: The UMAP plot before there is a UMAP.
UMAP_EMPTY = ("No UMAP yet.|Press UMAP under QC Features. It takes about "
              "45 s for 3,000 labels.")
#: The t-SNE plot before there is a t-SNE.
TSNE_EMPTY = ("No t-SNE yet.|Press tSNE under QC Features. It takes about "
              "20 s for 3,000 labels.")
#: A map plot while that map is computing.
MAP_BUSY_EMPTY = "Computing {map}…|The plot fills in when it is done."
#: The axes row on a violin plot.
VIOLIN_GROUPS = "{feat} by {by}"
#: A violin plot whose feature has no value on any active label.
VIOLIN_NO_VALUES = ("No values for {feat} on this patch.|Pick another "
                    "feature next to Feature.")
#: The axes row on a UMAP / t-SNE plot.
MAP_AXES = "{map} 1 vs {map} 2 · made from {input}"
#: The ghosts toggle's word (Step 4's is "Removed").
GHOSTS_WORD = "Ghosts"
#: The ghosts entry of the legend (plain text, never clickable).
GHOSTS_LEGEND = "ghosts"
#: The word for a label with no tier (Step 4's is "unmarked").
UNMARKED_WORD = "unassigned"
#: A cluster's legend word: ``{id}`` is the cluster id (C0, C1 ... as
#: Step 3 has always named them; Step 4's "G{n}" uses ``{n}`` = id + 1).
GROUP_WORD = "C{id}"
#: The noise cluster's legend word.
NOISE_WORD = "noise"

#: What Step 3 passes as ``QCPlotWidget(texts=...)``.
STEP3_PLOT_TEXTS = {
    "help": PLOT_HELP_LINES,
    "UMAP_EMPTY": UMAP_EMPTY,
    "TSNE_EMPTY": TSNE_EMPTY,
    "MAP_BUSY_EMPTY": MAP_BUSY_EMPTY,
    "VIOLIN_GROUPS": VIOLIN_GROUPS,
    "VIOLIN_NO_VALUES": VIOLIN_NO_VALUES,
    "MAP_AXES": MAP_AXES,
    "GHOSTS_WORD": GHOSTS_WORD,
    "GHOSTS_LEGEND": GHOSTS_LEGEND,
    "UNMARKED_WORD": UNMARKED_WORD,
    "GROUP_WORD": GROUP_WORD,
    "NOISE_WORD": NOISE_WORD,
}

# --- The plot's empty sentences (``set_empty_sentence``) ---

#: No patch is loaded (or a patch is loading).
PLOT_NO_PATCH = ("Load a patch to see its QC plot.|Click a patch in the "
                 "list on the left.")
#: A patch with no QC features yet.
PLOT_NO_FEATURES = ("No QC features for this patch yet.|Press Extract under "
                    "QC Features.")

# --- The see-line: "{n} labels · {g} ghosts · {sel} · coloured by {x}" ---

SEE_SEP = " · "
SEE_LABELS = "{n} labels"
SEE_LABELS_ONE = "1 label"
SEE_GHOSTS = "{n} ghosts"
SEE_GHOSTS_ONE = "1 ghost"
SEE_SELECTED = "{n} selected"
SEE_NONE_SELECTED = "none selected"
#: The whole see-line, least important part last (P2 audit MED-4: the
#: Colour by combo already says what the points are coloured by).  Built
#: from SEE_LABELS / the selected part / SEE_GHOSTS joined by SEE_SEP; in
#: the plural it reads exactly as this template.
SEE_LINE = "{n} labels · {sel} · {g} ghosts"
#: The see-line with no patch, with no features, and while a patch loads.
SEE_NO_PATCH = "No patch loaded."
SEE_NO_FEATURES = "No QC features yet."
SEE_LOADING = "Loading {patch}…"

# --- The status slot (``set_status(text, kind)``) ---

#: Kinds a status line can have (info grey, warn gold, error red).
STATUS_KINDS = ("info", "warn", "error")
STATUS_IDLE_HINT = "Lasso or click points to select."
#: The same hint on a violin plot (Lasso and Density are greyed there).
STATUS_IDLE_HINT_VIOLIN = "Click a violin to select its group."
STATUS_ISOLATED = "Viewer: {n} labels shown alone"
STATUS_ISOLATED_ONE = "Viewer: 1 label shown alone"
STATUS_ALL_SHOWN = "Viewer: showing every label"
STATUS_ISOLATION_FAILED = "Isolation failed — toggle Isolate off and retry."
STATUS_ASSIGNED = "Assigned {n} labels to {tier}."
STATUS_ASSIGNED_ONE = "Assigned 1 label to {tier}."
STATUS_UNDONE = "Undid the last assignment."
#: While the viewer is being recoloured for a selection / a clear.
STATUS_ISOLATING = "Isolating {n} labels…"
STATUS_ISOLATING_ONE = "Isolating 1 label…"
STATUS_RESTORING = "Restoring colours…"
#: A selection the viewer refuses while a refinement is staged (warn).
STATUS_FINISH_STAGED = ("Finish or reject the staged refinement before "
                        "isolating.")

# --- The lower area's tabs ---

TAB_HISTOGRAM = "Histogram"
TAB_SELECTION = "Selection"
#: The Selection tab with its count (a fixed-width slot, so a count never
#: moves its neighbour).
TAB_SELECTION_COUNT = "Selection · {n}"

# --- The Histogram view ---

HIST_TRIAGE_BY = "Triage by"
HIST_COLOUR_BARS_BY = "Colour bars by"
HIST_RESET_LINES = "Reset lines"
HIST_GATE = "Gate"
HIST_NO_DATA = "No QC data yet."
HIST_NO_VALUES = "No values for {feature} on this patch."
HIST_DRAG_NOTE = ("Dragging a line re-sorts every label that is not curated, "
                  "hand assignments included.")
HIST_ALL_CURATED = "All {n} labels are curated; the lines change nothing."
HIST_ALL_CURATED_ONE = "The only label is curated; the lines change nothing."

# --- The Selection view (the card) ---

CARD_SELECTED = "{n} labels selected"
CARD_SELECTED_ONE = "1 label selected"
CARD_NONE = ("No labels selected. Drag a lasso, click a point, click a "
             "colour in the legend, or tick Gate under Histogram (any "
             "feature but qc_score).")
CARD_MSG_LASSO = "Lasso: {n} labels."
CARD_MSG_LASSO_ONE = "Lasso: 1 label."
CARD_MSG_GATE = "Gate: {n} labels with {feature} between {lo} and {hi}."
CARD_MSG_GATE_ONE = "Gate: 1 label with {feature} between {lo} and {hi}."
CARD_MSG_LEGEND = "Legend: {tier}, {n} labels."
CARD_MSG_LEGEND_ONE = "Legend: {tier}, 1 label."
CARD_MSG_CLICK = "Click: label {id}."
CARD_MSG_ECHO = "From the Label Navigator or the viewer: {n} labels."
CARD_MSG_ECHO_ONE = "From the Label Navigator or the viewer: 1 label."
CARD_MSG_KEPT = "Kept from the last plot: {n} labels."
CARD_MSG_KEPT_ONE = "Kept from the last plot: 1 label."
#: A group picked on the violin plot (a violin or its legend colour).
CARD_MSG_VIOLIN = "Violin: {group}, {n} labels."
CARD_MSG_VIOLIN_ONE = "Violin: {group}, 1 label."
#: The card's warning when an Assign refused every selected label.
CARD_WARN_RECOMPUTING = ("All {n} selected labels are being recomputed; "
                         "none was assigned.")
CARD_WARN_RECOMPUTING_ONE = ("The selected label is being recomputed; it "
                             "was not assigned.")
#: Undo assign refused: the snapshot is of another table (P2 audit HIGH-1).
CARD_WARN_UNDO_STALE = ("Undo refused: the last tier change was made on "
                        "another patch.")
#: One chip / one part of the totals line: "curated 842".
CARD_TIER_COUNT = "{tier} {n}"
#: The totals line; ``{parts}`` is CARD_TIER_COUNT joined by SEE_SEP and
#: ``{n}`` the ghosts (``count_text(g, CARD_TOTALS, CARD_TOTALS_ONE_GHOST,
#: parts=...)``).
CARD_TOTALS = "All labels: {parts}, plus {n} ghosts"
CARD_TOTALS_ONE_GHOST = "All labels: {parts}, plus 1 ghost"
CARD_TOTALS_NO_GHOSTS = "All labels: {parts}"

# --- The action row (pinned under the lower area) ---

ACTION_ASSIGN_LABEL = "Assign {n} selected to"
#: The same with nothing selected (never "Assign 0 selected to").
ACTION_ASSIGN_LABEL_NONE = "Assign selected to"
ACTION_ASSIGN = "Assign"
ACTION_UNDO = "Undo assign"
ACTION_CLEAR = "✕ Clear"

# --- The left column with no patch (a run switch, Browse) ---

QC_SUMMARY_NONE = "No QC data"
CURATION_STATUS_NONE = "No active curation session."

# --- The see-line's tick ---

COLOURS_IN_VIEWER = "Colours in viewer"

# --- Help for the area's own controls (tooltips, one plain line each) ---

STEP3_PLOT_HELP = {
    "Histogram tab": "Show the histogram of one feature, with the tier "
                     "lines on qc_score.",
    "Selection tab": "Show what is selected: how many labels, their tiers, "
                     "and the totals for this patch.",
    "Divider": "Drag to give the plot or the area under it more room. "
               "Double-click to put it back.",
    "Triage by": "The feature the histogram shows. On qc_score its three "
                 "lines sort the labels into tiers.",
    "Colour bars by": "What each histogram bar is split by, for example "
                      "the QC tier.",
    "Reset lines": "Put the three tier lines at the quartiles of qc_score "
                   "on this patch and sort the labels again.",
    "Gate": "On any feature but qc_score: drag the band on the histogram "
            "to select the labels whose values fall inside it.",
    "Assign": "Give every selected label the tier picked in the list.",
    "Undo assign": "Take back the last tier change: an assignment, a line "
                   "drag or Reset lines.",
    "Clear": "Deselect all labels. Their tiers are not changed.",
    "Colours in viewer": "Paint the labels in the napari viewer with the "
                         "plot's colours. Off: napari's own label colours.",
    "Status": "What just happened: what the viewer shows, or the last "
              "assignment.",
    "Ghosts legend": "Ghosts are labels deleted or replaced in earlier "
                     "edits. They are counted here but cannot be selected.",
}


def count_text(n: int, many: str, one: str, **kw) -> str:
    """``one`` for exactly one, else ``many.format(n=n, **kw)``.

    >>> count_text(3, SEE_LABELS, SEE_LABELS_ONE)
    '3 labels'
    """
    if int(n) == 1:
        return one.format(**kw)
    return many.format(n=f"{int(n):,}", **kw)
