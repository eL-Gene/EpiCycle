"""Every sentence and help line of the Step 4 EDA Workbench.

Copied VERBATIM from the accepted Workbench mockup: its
``S`` object (sentences) and its ``H`` object (one help line per control).
``tests/segmentation/test_workbench_text_parity.py`` and
``test_workbench_help_parity.py`` read both objects out of the mockup file
and hold this module to them.  Step 4's own departures (``TOTALS_STEP4``,
``WB_LOCKED``, ``STOPPING``, ``LOG_MAP_ABANDONED``, ``PCA_TABLE_CAPTION``)
are deliberate and are listed in the parity test as intended.

A sentence that has a title and a body (the empty-map sentences) keeps the
mockup's ``|`` between them; :func:`split_title` takes it apart.
"""

from __future__ import annotations


#: The line under the dock title, always.
SEE_LINE = "{n} cells · {m} selected · coloured by {c}"
#: The same line with no selection.
SEE_LINE_NONE = "{n} cells · none selected · coloured by {c}"
#: Selection card, nothing selected.
SEL_NONE = "No cells selected. Drag a lasso, click a point, click a colour in the legend, or gate a feature."
#: Selection card title.
SEL_COUNT = "{m} cells selected"
#: Selection card title, one cell.
SEL_ONE = "1 cell selected (label {id})"
#: After Mark good / Mark bad.
MARKED = "Marked {m} cells {tier}. Maps are kept."
#: After Unmark.  Step 4 says "unmarked" everywhere (audit
#: X1-6): the mockup's "unassigned" is reworded, a recorded departure.
UNMARKED = "Returned {m} cells to unmarked."
#: A mark pressed with nothing selected.
MARK_NEEDS_SEL = "Select cells first, then mark them."
#: After Undo.
UNDONE = "Undid the last mark ({m} cells)."
#: Undo with an empty history.
UNDO_EMPTY = "Nothing to undo."
#: Under the selection card.
TOTALS = "All cells: good {g} · review {r} · bad {b} · unassigned {u}"
#: After a legend colour is clicked.
CAT_PICKED = "Selected every cell in “{cat}” ({m})."
#: After a lasso is released.
LASSO_DONE = "Lasso: {m} cells."
#: Lasso on while the plot is a violin.
LASSO_VIOLIN = "The lasso works on maps and scatters. On a violin, click a group to select it."
#: UMAP plot before a run.
UMAP_EMPTY = "No UMAP yet.|Press Compute UMAP on the Map tab. It takes about 45 s for 3,000 cells."
#: t-SNE plot before a run.
TSNE_EMPTY = "No t-SNE yet.|Press Compute t-SNE on the Map tab. It takes about 20 s for 3,000 cells."
#: Map plot while its run is going.
MAP_BUSY_EMPTY = "Computing {map}…|The plot fills in when it is done. You can keep marking cells meanwhile."
#: Busy strip.
BUSY = "Computing {map} · about {s} s"
#: After Stop.
STOPPED = "{map} stopped. The last map is kept."
#: Map tab result line.
MAP_DONE = "{map} ready: made from {input}."
#: Map tab result line after Compute PCA.
PCA_DONE = "PCA on {f} features: PC1 and PC2 explain {v}% of the variation."
#: Map tab, amber, after a change.
PCA_STALE = "The feature list or scaling changed. Press Compute PCA to update the map."
#: Map tab, amber, after a new PCA.
MAPS_OLD = "The UMAP and t-SNE were made from the previous PCA. Compute them again to match."
#: Compute PCA with fewer than two features.
PCA_TOO_FEW = "Tick at least two features to compute a PCA."
#: A second map pressed during a run.
BUSY_ONE = "Wait for {map} to finish, or press Stop."
#: Groups tab result line.
CLUSTER_DONE = "{k} groups from {method}, seed {seed}. The same seed gives the same groups."
#: Groups tab result line, HDBSCAN.
CLUSTER_DONE_HDB = "{k} groups and {noise} unassigned cells from HDBSCAN."
#: Cluster on UMAP with no UMAP.
CLUSTER_NEEDS_UMAP = "Compute a UMAP first, or group from PCA or the features."
#: Beside the greyed seed.
SEED_OFF = "HDBSCAN has no random start, so it needs no seed."
#: Colour by Group before a run (item text).
NO_GROUPS = "Press Cluster on the Groups tab to colour by group."
#: Under the feature table.
FEAT_COUNT = "{k} of {n} features used for the maps"
#: After a gate drag.
GATE_LINE = "Gate: {m} cells with {feat} between {a} and {b}."
#: Feature detail.
BELL_YES = "Bell-shaped: skew {s}, tail weight {k}."
#: Feature detail.
BELL_NO = "Not bell-shaped: skew {s}, tail weight {k}. Auto rescale picks {x}."
#: Feature detail, dim.
SHAPIRO = "Shapiro–Wilk p = {p}. With thousands of cells this test says “not normal” for almost any feature, so the Bell column uses skew and tail weight instead."
#: Viewer caption, sync on.
SYNC_ON = "The viewer uses the plot’s colours."
#: Viewer caption, sync off.
SYNC_OFF = "The viewer keeps napari’s own label colours."
#: Violin axes row.
VIOLIN_GROUPS = "{feat} by {by}"
#: Axes row on UMAP / t-SNE.
MAP_AXES = "{map} 1 vs {map} 2 · made from {input}"

# --- Step 4's own sentences ---

#: Under the selection card, Step 4's words (no review tier; 'unmarked').
TOTALS_STEP4 = "All cells: good {g} · bad {b} · unmarked {u}"
#: The selection card's message line while a Step 4 job other than a map runs.
WB_LOCKED = "A Step 4 job is running ({job}). Marking and maps wait until it ends."
#: B2 R1-6: the dock's window title (= the controller's dock name).
WB_WINDOW_TITLE = "EDA Workbench"
#: B2 R1-6: the feature detail panel's numbers line (``{xfrm}`` is
#: " (method)" or "").
WB_DETAIL_STATS = ("n={n}  mean={mean:.2f}  median={median:.2f}  "
                   "std={std:.2f}  min={min:.2f}  max={max:.2f}  "
                   "IQR={iqr:.2f}  skew={skew:.2f}  kurt={kurtosis:.2f}"
                   "{xfrm}")
#: B2 UI re-audit MED-1: the Features tab's words that were literals.
WB_NO_DESCRIPTION = "No description available."
WB_SKEW_TIP = "Fisher's skewness: {v:.3f}"
WB_SKEW_TIP_NONE = "Insufficient data"
WB_NORM_TIP_NONE = "Insufficient data for normality test"
WB_NORM_TIP_NORMAL = "{test}: p = {p:.4f} (can't reject normality)"
WB_NORM_TIP_NON = "{test}: p = {p:.4f} (significantly non-normal)"
WB_NORM_TEST_SHAPIRO = "Shapiro-Wilk"
WB_NORM_TEST_DAGOSTINO = "D'Agostino-Pearson"
COMPUTE_PCA_BTN = "Compute PCA"
COMPUTE_RESIDUAL_PCA_BTN = "Compute Residual PCA"
#: B2 K7: the Workbench shown outside Step 4 (napari's Window menu).
WB_STEP_INACTIVE = "Step 4 is not open. Go back to Step 4 to mark cells."
#: v2.12 audit LOW-2: a Workbench rebuilt after napari's X, when the closed
#: one held maps or ticks that were never saved (the card's message line).
WB_REBUILT_FROM_SAVED = ("Reopened as last saved; maps made since the last "
                         "Save were not kept.")
#: The busy strip after Stop, until the orphaned fit ends.
STOPPING = "Stopping {map}… a new map can start when it has."
#: The log line when a change of the cells stops a running map.
LOG_MAP_ABANDONED = "{map} was stopped because the cells changed."
#: Under the PCA loadings table.
PCA_TABLE_CAPTION = (
    "Every ticked feature against every component, sorted by PC1. Green "
    "pushes a cell up the component, red down; the bar over each column is "
    "its share of the variation."
)

# --- B3b: words the behaviour needs beyond the tables ---

#: One chip of the selection card (a coloured dot, the tier, its count).
CHIP_HTML = ('<span style="color:{colour};">&#9679;</span>&nbsp;{tier} '
             '<b style="color:#dddddd;">{n}</b>')
CHIP_SEP = " &nbsp;&nbsp; "
#: One swatch of the clickable Groups legend.
GROUP_SWATCH_HTML = ('<a href="group://{cid}" style="color:rgb({r},{g},{b});'
                     ' text-decoration:none;">&#9679;</a>&nbsp;'
                     '<a href="group://{cid}" style="color:#dddddd;'
                     ' text-decoration:none;">{word}</a> '
                     '<span style="color:#888888;">{n}</span>')
#: What a map was made from (MAP_DONE / MAP_AXES ``{input}``), the
#: mockup's words.
INPUT_PCA = "PCA ({n} components)"
INPUT_RESIDUAL_PCA = "residual PCA ({n} components)"
INPUT_FEATURES = "the rescaled features"
INPUT_UMAP = "the UMAP"
#: The map words of BUSY / STOPPED / BUSY_ONE / MAP_BUSY_EMPTY.
MAP_WORDS = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE",
             "cluster": "Groups"}
#: The method words of CLUSTER_DONE.
CLUSTER_METHOD_WORDS = {"leiden": "Leiden", "hdbscan": "HDBSCAN",
                        "kmeans": "K-means", "gmm": "GMM"}
#: A map whose computation raised (the log carries the reason).
MAP_FAILED = "{map} could not be computed. The log has the reason."
#: A map run with too few cells to lay out.
MAP_TOO_FEW_CELLS = "{map} needs at least 3 cells."
#: The words WB_LOCKED names a running Step 4 job by.
LOCK_JOB_WORDS = {
    "_predict_running": "Predict", "_reconstruct_running": "Reconstruct",
    "_load_in_flight": "Load", "_preview_in_flight": "Load",
    "_measure_running": "Measure", "_smooth_running": "Smooth",
    "_save_in_flight": "Save", "_export_running": "Export",
    "_mesh_running": "Export",
}
#: The singulars the mockup's table leaves out (shots: "Marked 1 cells").
MARKED_ONE = "Marked 1 cell {tier}. Maps are kept."
UNMARKED_ONE = "Returned 1 cell to unmarked."
UNDONE_ONE = "Undid the last mark (1 cell)."
LASSO_DONE_ONE = "Lasso: 1 cell."
#: BUSY on the first UMAP of a session (numba compiles; audit X1-5).
BUSY_WARMUP = ("Computing {map} · warming up, the first map of a session "
               "can take a minute")
#: BUSY when the estimate is under 5 s.
BUSY_FEW = "Computing {map} · a few seconds"
#: The legend's words (passed to the plot through ``texts=``, LOW-3).
UNMARKED_WORD = "unmarked"
GROUP_WORD = "G{n}"
NOISE_WORD = "noise"
#: MAP_DONE when the landed map did not take the plot (re-audit LOW-5).
MAP_DONE_PICK = ("{map} ready: made from {input}. Pick {map} in the plot "
                 "type to see it.")
#: A map from the features with no feature ticked (audit E1-4).
MAP_NEEDS_FEATURES = "Tick at least one feature to compute {map} from the features."
#: The seconds BUSY quotes for 3,000 cells (the mockup's figures;
#: scaled linearly by the cell count, rounded to 5 s).
BUSY_SECONDS_3000 = {"umap": 45, "tsne": 20, "pca": 5, "cluster": 5}


def busy_text(kind: str, n_cells: int, rate=None, warmed=False) -> str:
    """The busy strip's line (audit X1-5): the first UMAP of a session
    says BUSY_WARMUP (numba compiles, no number is honest); that run's
    time is never a rate (re-audit MED-1).  Later runs use the last
    measured seconds per cell for that kind, else the mockup's 3,000-cell
    figure scaled; under 5 s, BUSY_FEW."""
    word = MAP_WORDS.get(kind, kind)
    if rate is None:
        if kind == "umap" and not warmed:
            return BUSY_WARMUP.format(map=word)
        raw = BUSY_SECONDS_3000.get(kind, 5) * max(0, int(n_cells)) / 3000.0
    else:
        raw = float(rate) * max(0, int(n_cells))
    if raw < 5:
        return BUSY_FEW.format(map=word)
    return BUSY.format(map=word, s=int(5 * round(raw / 5.0)))


def busy_seconds(kind: str, n_cells: int) -> int:
    """BUSY's ``{s}``: the 3,000-cell figure scaled by the cell count,
    rounded to 5 s, at least 5 s."""
    base = BUSY_SECONDS_3000.get(kind, 5)
    raw = base * max(0, int(n_cells)) / 3000.0
    return max(5, int(5 * round(raw / 5.0)))


# --- Words the mockup draws as plain labels (headings, tabs, buttons) ---

TAB_FEATURES = "Features"
TAB_MAP = "Map"
TAB_GROUPS = "Groups"
FEATURES_TITLE = "Features for the maps"
FEATURES_DESC = ("Ticked features feed PCA, UMAP, t-SNE and grouping. "
                 "Click a row for its histogram.")
PCA_RESULTS_TITLE = "PCA results"
MAP_PCA_TITLE = "PCA"
MAP_PCA_DESC = "the map every other map starts from"
MAP_UMAP_TITLE = "UMAP"
MAP_UMAP_DESC = "about 45 s for 3,000 cells"
MAP_TSNE_TITLE = "t-SNE"
MAP_TSNE_DESC = "about 20 s for 3,000 cells"
GROUPS_TITLE = "Find groups"
GROUPS_DESC = "colours the cells by group; marks are not changed"
GROUPS_LEGEND_HINT = "Click a group to select it."
COLOURS_IN_VIEWER = "Colours in viewer"
CATEGORY_HEADER = "▾ {cat}  {on} of {n} used"
CHANNEL_GROUP = "{channel} channel"
#: A folded category header (the caret turns; addition, W1-3).
CATEGORY_HEADER_SHUT = "▸ {cat}  {on} of {n} used"
#: A category header's tooltip (addition, W1-3).
CATEGORY_HEADER_TIP = ("Click to fold or unfold this group. Ctrl+click to "
                       "tick or untick every feature in it.")
#: One line under PCA_DONE after a residual PCA (addition, W1-12).
PCA_EXCLUDED = "{k} regress-out features were left out of the PCA input."
#: The feature table's column headers (owner: Bell).
FEATURE_COLUMNS = ("", "Feature", "Shape", "Skew", "Bell", "Rescale")


#: One help line per control, verbatim from the mockup's ``H`` table.
WORKBENCH_HELP_LINES = {
    "Plot type": "Choose how to lay out the cells: two features, the PCA map, the UMAP or t-SNE map, or a violin per group.",
    "Lasso": "Drag a loop around cells to select them. Click without dragging to select one cell.",
    "Density": "Draw contour lines where cells are packed densely.",
    "Removed": "Show cells removed in earlier edits as hollow grey points. They cannot be selected.",
    "Colour by": "Colour the points by a tier, a group, border cells or any measured feature.",
    "Scale": "How feature values map to colours: Linear, Log, or Rank (even spread, the default).",
    "Colours": "The colour scale used for a feature.",
    "Colours in viewer": "Paint the cells in the napari viewer with the same colours as the plot.",
    "Mark good": "Mark every selected cell good. Maps and groups are kept.",
    "Mark bad": "Mark every selected cell bad. Maps and groups are kept.",
    "Unmark": "Return the selected cells to unmarked.",
    "Undo": "Take back the last mark.",
    "Clear": "Deselect all cells. Marks are not changed.",
    "Shape": "A small histogram of the feature after its rescale.",
    "Skew": "How lopsided the values are. Arrow up-right: a long tail of large values; down-right: a long tail of small values.",
    "Bell": "Filled dot: the values look bell-shaped (small skew and ordinary tails).",
    "Rescale": "How the values are transformed before PCA, UMAP and grouping. Auto uses log for a long right tail and none for bounded or symmetric features.",
    "Auto rescale": "Set every feature’s rescale by the skew rule: log for right-skewed, none for bounded or symmetric.",
    "Defaults": "Tick the default feature set and set every rescale back to auto.",
    "Gate": "Drag a range on the histogram to select the cells whose value falls inside it.",
    "Colour plot by this": "Colour the plot by this feature.",
    "Method (Map)": "PCA, or residual PCA, which first removes the effect of cell size from every feature.",
    "Components": "How many principal components to keep. UMAP, t-SNE and grouping start from these.",
    "Scaling": "Standard centres on the mean and divides by the spread; robust uses the median and MAD and is less pulled by outliers.",
    "Clip extremes": "Cap values more than this many spreads from the centre before PCA (winsorize).",
    "Neighbours": "How many nearby cells UMAP looks at. Larger values show broader structure.",
    "Spacing": "How tightly UMAP packs similar cells (min_dist). Smaller values give tighter clumps.",
    "Start from": "Compute from the PCA components or straight from the rescaled features.",
    "Distance": "How the difference between two cells is measured.",
    "Perplexity": "Roughly how many neighbours t-SNE balances for each cell.",
    "Method (Groups)": "Leiden finds communities in a neighbour graph; HDBSCAN finds dense clumps and leaves stragglers unassigned; K-means and GMM split into a set number of groups.",
    "Resolution": "Leiden only. Higher values give more, smaller groups.",
    "Number of groups": "K-means and GMM only.",
    "Smallest group": "HDBSCAN only. Clumps smaller than this are left unassigned.",
    "Seed": "The random start. The same seed and settings always give the same groups.",
    "Divider": "Drag up or down to give the plot or the tabs more room. Double-click to reset.",
}


def split_title(sentence: str) -> tuple[str, str]:
    """``"Title|Body"`` -> ``("Title", "Body")``; no ``|`` -> ``(sentence, "")``."""
    title, _, body = sentence.partition("|")
    return title, body


#: Owner: a feature is bell-shaped when |skew| < 0.5 and
#: |excess kurtosis| < 1.
BELL_MAX_ABS_SKEW = 0.5
BELL_MAX_ABS_KURT = 1.0


def is_bell(skew, kurt) -> bool:
    """The Bell column's rule; NaN is never bell-shaped."""
    try:
        s, k = float(skew), float(kurt)
    except (TypeError, ValueError):
        return False
    if s != s or k != k:  # NaN
        return False
    return abs(s) < BELL_MAX_ABS_SKEW and abs(k) < BELL_MAX_ABS_KURT
