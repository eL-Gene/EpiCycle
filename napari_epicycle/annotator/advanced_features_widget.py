"""Advanced Features dock widget for EpiCycle multiview annotator.

Provides tabbed access to segmentation workflows ported from
napari-multiview-annotator: SAM-assisted annotation, guided segmentation
(membrane walker), scribble segmentation, histogram-based segmentation,
and assisted split (seeded watershed).

Workflows are currently UI stubs — backend adapters will be connected in
future phases. Assisted Split (K13.2) is the first fully wired tool.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QSize, Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from functools import partial as _partial
from pathlib import Path as _Path

from napari_epicycle.ui_shell.theme import (
    DESTRUCTIVE_BTN_STYLE,
    DIM_LABEL_STYLE,
    INPUT_STYLE,
    LIST_STYLE,
    PANEL_ADV_LABEL_COLUMN,
    PANEL_BTN_STYLE,
    PANEL_CHECKBOX_STYLE,
    PANEL_CHEVRON_STYLE,
    PANEL_CHIP_STYLE,
    PANEL_ERROR_STYLE,
    PANEL_FOOTER_OBJECT,
    PANEL_FOOTER_STYLE,
    PANEL_GOLD,
    PANEL_GRIP_H,
    PANEL_GRIP_STYLE,
    PANEL_HELP_STYLE,
    PANEL_INPUT_STYLE,
    PANEL_LABEL_COLUMN,
    PANEL_LIST_STYLE,
    PANEL_PRIMARY_BTN_STYLE,
    PANEL_ROW_LABEL_STYLE,
    PANEL_STATUS_PILL_OBJECT,
    PANEL_STATUS_PILL_STYLE,
    PANEL_WARN_STYLE,
    PANEL_GOLD_HOVER,
    PRIMARY_BTN_STYLE_CYAN,
    PRIMARY_BTN_STYLE_GREEN,
    PRIMARY_BTN_STYLE_TEAL,
    PanelRowDelegate,
    SECONDARY_BTN_STYLE,
    SECTION_HEADER_STYLE,
    SEPARATOR_LINE_STYLE,
    STATUS_LABEL_STYLE,
    TAB_WIDGET_STYLE,
    help_button as _help_button,
    make_panel_card,
    panel_card_header,
    panel_form_label,
)

#: The Advanced Features dock had no help system before the guided Split; the Split
#: tab's `?` is its first entry.  Same guides dir the
#: 3D annotator's own help buttons use.
from napari_epicycle.ui_shell._guides import guides_dir as _guides_dir
from napari_epicycle.annotator import _reshape_text as _RT
_GUIDES_DIR = _guides_dir()

# Refinement workflow button color palette (2026-05-16) — applied
# consistently to every Preview/Accept/Reject button group across
# every section of this widget.  Mirrors the merge dialog and the
# slim-utility-panel buttons in Curation3DAnnotator for a unified
# refinement visual language.  See ui_shell/theme.py for the palette.
_REFINEMENT_PREVIEW_STYLE = PRIMARY_BTN_STYLE_TEAL   # primary investigation
_REFINEMENT_ACCEPT_STYLE = PRIMARY_BTN_STYLE_GREEN   # commit
_REFINEMENT_REJECT_STYLE = DESTRUCTIVE_BTN_STYLE     # undo to snapshot

# ---------------------------------------------------------------------------
# The two "guided Split" pickers.
#
# Both default to TODAY'S behaviour, so a user who never opens the collapsed
# row sees no change at all.  The orchestrator branches on these
# indices, and the tests assert the legacy path is untouched at the defaults,
# so they are module constants rather than magic numbers.
# ---------------------------------------------------------------------------
SEEDS_FROM_PAINTED = 0        # today: hand-painted ids on the Refine layer
SEEDS_FROM_DETECTED = 1       # today: auto EDT peaks (legacy _detect_seeds)
SEEDS_FROM_RUN_LABELS = 2     # NEW: nuclei labels from another run of this patch
SEEDS_FROM_EXTERNAL_MASK = 3  # NEW: an external TIFF
SEEDS_FROM_IMAGE = 4          # Brick 2a: find them in an image channel

#: Seed sources that need a trusted guide mask loaded first.
GUIDED_SEED_SOURCES = frozenset({SEEDS_FROM_RUN_LABELS, SEEDS_FROM_EXTERNAL_MASK})

#: Seed sources that reach the ENGINE rather than the legacy code.  This
#: is a ROUTING set (`is_guided_split()` reads it) and it does NOT decide
#: which buttons a card shows any more -- see `FIND_MARKER_SOURCES` and
#: `DETECT_SEED_SOURCES` below ("one finder per card").
IMAGE_DETECTOR_SOURCES = GUIDED_SEED_SOURCES | {SEEDS_FROM_IMAGE}

#: The sources whose markers come from a FIND.  Index 1
#: joined them in Brick U1: "Shape thickness peaks" used to show the
#: painting rows and then silently discard everything painted, because
#: the engine's `distance` detector never looks at the Refine layer
#: (bug B).  It now finds its markers and MATERIALISES them into
#: the Refine layer, exactly like a painted one.
FIND_MARKER_SOURCES = IMAGE_DETECTOR_SOURCES | {SEEDS_FROM_DETECTED}

#: ...and "Detect Seeds" is source 0's finder, and only source 0's: one
#: finder per card.  Index 1 has Find markers instead.
DETECT_SEED_SOURCES = frozenset({SEEDS_FROM_PAINTED})

#: The sources whose found markers are written INTO the Refine layer
#: instead of being shown in their own `guide_markers` layer, because
#: the cut then reads them back as painted ids.  Drawing the same
#: markers twice, in two colormaps, is worse than not drawing them.
#:
#: **EVERY found source, since U1b**.  It used to be index 1
#: alone, and the owner found the consequence on real data: the markers
#: of sources 2, 3 and 4 lived in the unmanaged `guide_markers` layer,
#: which only the MAIN viewer holds -- so the XY / XZ / YZ side views,
#: which is where the markers are painted and checked, showed nothing.
#: The Refine layer is mirrored into all three, so putting the markers
#: there is the whole fix.  `guide_markers` is not created at all any
#: more; the registry stays for `guide_walls` and the loaded mask.
MATERIALISED_MARKER_SOURCES = FIND_MARKER_SOURCES

#: The image-reading detectors, in the combo's order (commonest first).
#: ORDER IS THE WIRE FORMAT -- append only.
SEED_DETECTORS = (
    "peaks", "hmax", "blobs_log", "blobs_dog", "threshcc", "hmin",
)

#: Their labels, same index order (verbatim).
SEED_DETECTOR_LABELS = (
    "Nucleus peaks",
    "Prominent peaks",
    "Round-blob finder",
    "Round-blob finder (fast)",
    "Bright-object centres",
    "Membrane-enclosed interiors",
)

WALLS_FROM_SHAPE = 0          # today: geometry only
WALLS_FROM_MEMBRANE = 1       # NEW: a membrane image channel

#: Every wall kind the engine has, in the combo's order.  Index 0 and 1
#: are today's two; 2-8 are APPENDED so no existing index moves.
WALL_KINDS = (
    "distance",     # 0 - shape only (today)
    "membrane",     # 1 - membrane signal (today)
    "hybrid",       # 2
    "gradient",     # 3
    "intensity",    # 4
    "frangi",       # 5
    "meijering",    # 6
    "skeleton",     # 7
    "none",         # 8
)

#: The wall kinds that need an image channel (and that the arbiter may
#: weigh).  Mirrors ``reshape.spec.IMAGE_DERIVED_WALLS`` -- kept here as
#: a tuple of INDICES so the widget never imports the engine.
IMAGE_DERIVED_WALL_INDICES = frozenset(
    i for i, kind in enumerate(WALL_KINDS)
    if kind in ("membrane", "intensity", "frangi", "meijering",
                "gradient", "hybrid")
)

#: Brick U3: the legend under "Show what the tool sees".
#: The picture is the same greyscale whatever built it and NOTHING in it
#: says which, so the sentence is keyed on the wall kind.
#:
#: FOUR sentences, not two (fix pass M1), because "shape" is not one
#: physics.  Read off ``reshape/walls.py``:
#:
#: * ``distance`` ("Pinch points") is ``norm(-EDT(region))`` -- bright
#:   where the EDT is LOW, i.e. where the object is THIN.  Calling that
#:   "the object's own thickness" is the picture upside down: the user
#:   looks for the bright band and it is the WAIST, not the bulk.
#: * ``skeleton`` ("Between the spines") is ``norm(EDT(~skeleton))`` --
#:   bright FAR FROM the medial axis, so the spines of two merged objects
#:   are the DARK lines and the bright sheet between them is the cut.
#:   The opposite reading of the same greyscale.
#: * ``none`` ("No wall (midline only)") is literally ``zeros`` -- a flat
#:   picture, and a user told "bright = thin" would hunt for a bright
#:   band that cannot exist.
#: * every image kind is evidence out of the data, and that one sentence
#:   was already right.
WALLS_LEGEND_IMAGE = "Bright = where the tool thinks an edge is."
WALLS_LEGEND_PINCH = (
    "The object's own shape, not the image. Bright = thin - the pinch "
    "the cut goes through."
)
WALLS_LEGEND_SPINES = (
    "Bright = far from the object's spine - the valley between two "
    "merged objects. The spines are dark."
)
WALLS_LEGEND_FLAT = (
    "Nothing to see - this cut follows no wall, only the midline "
    "between markers."
)

#: ...and its two tooltips.  This one is deliberately NOT a ``HELP_LINES``
#: entry: an entry there is rendered as a grey line on the page, and with
#: this tick on it review-with-markers-open measures 749 px against a
#: 753 px ceiling (re-measured Brick U3).  A permanent
#: grey line is 14 px of those 4.  Same reasoning, and the same
#: precedent, as ``split_about_btn`` ('s amendment); what the user
#: reads on the page is the LEGEND, which is keyed on the wall kind and
#: is on screen only while the box is ticked.
WALLS_TOGGLE_HELP = (
    "Adds the wall picture the cut followed as a layer, so you can see "
    "why it cut where it did. Remove the tick to take it away."
)
WALLS_TOGGLE_DISABLED = "Preview first - there is nothing to show yet."

#: ...and what it says on the LEGACY path (fix pass H4).  Hand-painted
#: markers plus "Pinch points (shape)" never build an engine result, so
#: there is no elevation to show -- but the user got there by pressing
#: Preview, so "Preview first" is exactly wrong, and a grey tick with no
#: reason reads as a broken control.  This names the two knobs that would
#: give them a picture.
WALLS_TOGGLE_MANUAL = (
    "Only a wall-based cut makes a picture. Pick a marker finder, or a "
    "wall kind other than 'Pinch points (shape)'."
)

#: What the grey tick says after the byte budget refused the picture
#: (fix pass H5).  It points AT the legend, because Qt shows no tooltip
#: on a disabled widget and the sentence the user can actually read is
#: the one in the legend's slot.
WALLS_TOGGLE_TOO_LARGE = "Too large to show - see the line below."

#: Wall kinds the ``distance`` (shape-thickness) marker detector cannot
#: work against: it reads the WALL elevation, so a flat one finds nothing
#: and a medial-axis one is a different detector than its label says
#: (measured on a 10x20x20 box).
SHAPE_MARKER_FORBIDDEN_WALLS = frozenset(
    i for i, kind in enumerate(WALL_KINDS) if kind in ("none", "skeleton")
)

#: The picker labels, in index order.  Kept here so the orchestrator can quote
#: them back to the user in a message without reading the widget.
SEEDS_FROM_LABELS = (
    "Hand-painted markers",
    "Shape thickness peaks",
    "Markers from another run of this patch",
    "Markers from a mask file",
    "Find them in an image channel",
)

#: The same choices in as few words as fit the collapsed S3 line at
#: the 340 px dock.  Presentation only -- the INDEX is the wire format.
SEEDS_FROM_SHORT = (
    "painted", "shape", "run mask", "file mask", "image",
)

WALLS_FROM_LABELS = (
    "Pinch points (shape)",
    "Membrane sheets",
    # Not "Membrane plus shape": it is the sheet-ridge FILTER plus the
    # shape, and the name has to say which membrane thing it means --
    # it is one of the two deliberate carve-outs from the "no membrane
    # in an action" rule (M6).
    "Membrane sheets plus shape",
    "Intensity edges",
    "Raw brightness",
    "Vessel ridges",
    "Thin-faint ridges",
    "Between the spines",
    "No wall (midline only)",
)

WALLS_FROM_SHORT = (
    "shape", "membrane", "membrane+shape", "edges", "brightness",
    "vessels", "thin ridges", "spines", "none",
)

#: Engine solver names, in the order the Advanced combo lists them.
SPLIT_SOLVERS = ("watershed", "random_walker", "label_diffusion")

#: The "Solve with" row's labels, in the SAME index order as SPLIT_SOLVERS
#: ("ship this").  Item ORDER is the wire format and item TEXT is
#: presentation, so relabelling here is safe by construction -- and these
#: say what the solver does to the cut rather than naming the algorithm.
SPLIT_SOLVER_LABELS = (
    "Sharp cut (watershed)",
    "Soft cut (random walker)",
    # the wall channel can be ANY channel, so every ACTION stops
    # naming the membrane.  Only the wall TYPE "Membrane sheets" keeps
    # the word -- it names the sheet-ridge filter, not the channel.
    "Settle onto the wall signal (label diffusion)",
)

# ---------------------------------------------------------------------------
# The REFINE tab.  Everything below is read BY
# INDEX -- item order is the wire format, item text is presentation -- and
# every one of these tuples is a MODULE constant so the orchestrator and the
# tests read the same object the combo was filled from.
# ---------------------------------------------------------------------------

#: ``(index, combo text, engine key)``, in the combo's order.  ONE source of
#: order for the six things that used to hold it separately: the
#: combo's ``addItems``, the stack's page order, ``_REFINE_METHOD_DESCRIPTIONS``,
#: the annotator's dispatch dict, its ``method_label`` twin, and the two
#: contour branches that collapse into one.
#:
#: The reorder is safe HERE and nowhere else: the index is not persisted
#: anywhere -- ``batch_refine_recipes.json`` carries its own three-method
#: vocabulary (``batch_refine.py``'s ``_VALID_METHODS``) and never names a
#: Refine combo index.
REFINE_METHOD_KEYS = (
    (0, "Smooth (signed distance)", "sdf"),
    (1, "Active contour", "contour"),
    (2, "Dilate/Erode", "dilate_erode"),
    (3, "Guided Grow", "guided_grow"),
    (4, "Fill Holes", "fill_holes"),
    (5, "Flood Fill", "flood_fill"),
    (6, "Settle onto the wall signal", "settle"),
)

#: engine key -> combo index, derived so the two can never disagree.
REFINE_METHOD_INDEX = {key: idx for idx, _t, key in REFINE_METHOD_KEYS}

#: Refine recipes.  The button's
#: tooltip, the footer's idle line, and the rule that gives the menu button
#: Qt's own arrow on the right (no "▾" in the text) with room for it.
REFINE_RECIPES_TIP = (
    "Save the staged steps as a recipe, or run a saved recipe on the "
    "selected object. The recipe uses the channel picked on this page."
)
REFINE_FOOTER_IDLE = "Nothing is staged yet. Saved recipes are under Recipes."
_RECIPES_MENU_INDICATOR_STYLE = (
    "QPushButton { padding-right: 18px; }"
    "QPushButton::menu-indicator {"
    "  subcontrol-origin: padding; subcontrol-position: center right;"
    "  right: 5px;"
    "}"
)


class _ElideOnResize(QObject):
    """Re-elide the Refine stage line whenever it is shown or resized."""

    def __init__(self, fn, parent):
        super().__init__(parent)
        self._fn = fn

    def eventFilter(self, obj, event):  # noqa: N802 - Qt API
        if event.type() in (QEvent.Resize, QEvent.Show):
            try:
                self._fn()
            except Exception:
                pass
        return False

#: Guided Grow's reason, when the patch has no probability map.
#: The ITEM is greyed before any click -- a refusal that fires after the
#: click teaches nothing -- and, because Qt shows no tooltip on a disabled
#: item, the sentence is also put ON SCREEN under the method combo.
REFINE_GUIDED_GROW_NO_PROB = (
    "Guided Grow needs the reconstruction's probability map, and this "
    "patch has none."
)

#: ...and the short form the greyed item itself carries in the list.
REFINE_GUIDED_GROW_ITEM_NO_PROB = "Guided Grow (needs a probability map)"

#: The card sentence, shown where "Local contrast" is hidden.
REFINE_NO_CLAHE_SENTENCE = (
    "Local contrast is not used when the tool is looking for walls."
)

#: The Active contour's two modes, read BY INDEX: the order is
#: `refine.CONTOUR_MODES` ("snap", "brightness"), the text is presentation.
REFINE_CONTOUR_MODE_LABELS = ("Snap to edges", "Inside vs outside brightness")

#: Add's REVIEW hand-off of every added cell to Batch Refine
#:.  The same arrow glyph as "Refine ▸".
#: S1's text colour follows what S1 MEANS: a "warn" --
#: a blocking reason, a transient refusal, TICK_ONE -- is drawn amber (the
#: text colour of PANEL_WARN_STYLE), everything else in the page's dim
#: grey.  Colour only: the same font size, so S1's height never moves.
ADD_S1_STYLE = DIM_LABEL_STYLE
#: built from the theme's gold (A-L5, the A1-A4 audit): the same rendered
#: colour as PANEL_WARN_STYLE's text, one source
ADD_S1_WARN_STYLE = "color: {}; font-size: 10px;".format(PANEL_GOLD_HOVER)

ADD_REFINE_ALL_BTN = "Refine all ▸"
ADD_REFINE_ALL_TIP = (
    "Open Batch Refine on every cell added in this session — run a saved "
    "Refine recipe on all of them."
)


# ---------------------------------------------------------------------------
# The Add page.  Vocabulary read BY INDEX; new
# items APPEND.
# ---------------------------------------------------------------------------

#: The wall kinds the Add Cells preset offers, in the combo's order.  A
#: literal copy of ``reshape.spec.ADD_WALL_KINDS`` (IMAGE_DERIVED_WALLS minus
#: hybrid), ordered; the widget never imports the engine for it, and
#: ``test_add_presets`` pins the parity.
#: what the Brightness threshold box shows while its value is not known yet
#: (its value space changed and the next Auto has not landed -- B4)
ADD_THRESHOLD_PENDING_TEXT = "—"
ADD_WALL_KINDS = ("membrane", "gradient", "intensity", "frangi", "meijering")

#: One plain sentence per Add control, set as its TOOLTIP.  A
#: separate table from ``HELP_LINES`` on purpose: that table's contract
#: (``test_helper_lines``) is "every entry is rendered as a grey line under
#: its control", and one line per Add control would put the page far over
#: its 3.1-screen ceiling.  The on-screen reasons live in the page's help
#: slots (S1, the Markers help line, the Advanced help slot, the walls
#: legend); ``test_add_help_parity`` holds this table to every control.
#: Tooltip-only exceptions, named: the preset combo's item tooltips and
#: the six sliders inside Numbers....
from packages.epicycle_annotator.add_preview import (  # noqa: E402
    ADD_CELLS_PRESET_MAX_VOXELS as _AP_ADD_CELLS_PRESET_MAX_VOXELS,
    ADD_CLEAR_TIP as _AP_ADD_CLEAR_TIP,
    ADD_HELP_BRIDGE as _AP_ADD_HELP_BRIDGE,
    ADD_HELP_DARK_BELOW as _AP_ADD_HELP_DARK_BELOW,
    ADD_HELP_GROW as _AP_ADD_HELP_GROW,
    ADD_HELP_GROWTH_LIMIT as _AP_ADD_HELP_GROWTH_LIMIT,
    ADD_HELP_INCLUDE_MEMBRANE as _AP_ADD_HELP_INCLUDE_MEMBRANE,
    ADD_HELP_LIVE_ALWAYS as _AP_ADD_HELP_LIVE_ALWAYS,
    ADD_HELP_LIVE_TICK as _AP_ADD_HELP_LIVE_TICK,
    ADD_HELP_LOOK_FOR as _AP_ADD_HELP_LOOK_FOR,
    ADD_HELP_NUC_OUTLINE as _AP_ADD_HELP_NUC_OUTLINE,
    ADD_HELP_WALL_COST as _AP_ADD_HELP_WALL_COST,
    ADD_MARKER_SPACING_DEFAULT as _AP_ADD_MARKER_SPACING_DEFAULT,
    ADD_HELP_NUC_EDGE as _AP_ADD_HELP_NUC_EDGE,
    ADD_HELP_NUC_RADIUS as _AP_ADD_HELP_NUC_RADIUS,
    ADD_HELP_TWO_GAP as _AP_ADD_HELP_TWO_GAP,
    ADD_WHOLE_TIP as _AP_ADD_WHOLE_TIP,
    voxel_limit_words as _AP_voxel_limit_words,
)

ADD_HELP_LINES = {
    "add_preset_combo": (
        "What you are adding. Nuclei finds bright objects; Cells grows one "
        "object from each marker until it meets a wall."),
    "add_start_btn": (
        "Drops a box where the crosshair is. Scroll the side views to the "
        "empty space first."),
    # what they do -- plain help: the two are never grey
    "add_whole_btn": _AP_ADD_WHOLE_TIP,
    "add_size_whole_btn": _AP_ADD_WHOLE_TIP,
    "add_look_for_combo": _AP_ADD_HELP_LOOK_FOR,
    "add_nuc_radius_sb": _AP_ADD_HELP_NUC_RADIUS,
    "add_nuc_edge_sb": _AP_ADD_HELP_NUC_EDGE,
    "add_two_gap_sb": _AP_ADD_HELP_TWO_GAP,
    "add_size_s_btn": "Resize the box to about an eighth of the patch.",
    "add_size_m_btn": "Resize the box to about a quarter of the patch.",
    "add_size_l_btn": (
        "Resize the box to about half of the patch (for cells, at most 4 "
        "million voxels)."),
    "add_numbers_btn": "Type the box's centre and size, one number per axis.",
    # the button shows its glyph alone; the word lives here
    "add_recenter_btn": (
        "Recenter: move the box so its centre is where the crosshair is."),
    "add_marker_source_combo": (
        "Where each new object grows from: another run's objects, nuclei "
        "in a second channel, or your clicks."),
    "add_nuclei_channel_combo": "The channel the nuclei are bright in.",
    "add_nuclei_threshold_sb": (
        "Voxels brighter than this count as nucleus. Type a value, or press "
        "Auto."),
    "add_nuclei_auto_btn": (
        "The automatic split between background and nuclei in this box."),
    "add_place_markers_btn": (
        "While this is on, a click inside the box in a side view places a "
        "marker and starts a new object. Shift + click adds to the current "
        "object, Alt + click picks an object, Ctrl + click removes a "
        "marker, Ctrl + Z takes back the last one and Esc leaves no object "
        "current."),
    "add_click_undo_btn": (
        "Removes the last marker placed (Ctrl + Z in a side view)."),
    "add_click_clear_btn": "Removes every marker in this box.",
    "add_click_list": (
        "Each object has a row in its colour, with its seed count when it "
        "grows from several markers. Click a row to make that object "
        "current; untick one and the next Preview grows nothing from it."),
    "add_wall_channel_combo": "The channel the membrane is bright in.",
    "add_prep_zcorrect_cb": "Lifts deep, dim slices to match the shallow ones.",
    "add_prep_bgsub_cb": "Takes away smooth background glow.",
    "add_prep_lift_cb": "Brightens faint signal more than bright signal.",
    "add_prep_denoise_cb": "Removes salt-and-pepper noise.",
    "add_wall_kind_combo": (
        "What the edge between two cells looks like in the membrane "
        "channel."),
    "add_channel_combo": "The channel the nuclei are bright in.",
    "add_clahe_cb": (
        "Evens out bright and dim regions of the box before the threshold. "
        "Off by default."),
    "add_threshold_sb": (
        "Voxels at or above this count as object. Type a value or drag the "
        "red line."),
    "add_otsu_btn": (
        "Puts the threshold at the automatic split between background and "
        "the bright objects."),
    "add_preview_btn": (
        "Finds the new objects in the box. Nothing is written until "
        "Accept."),
    "add_stop_btn": "Stops the preview. Nothing from this box is added.",
    "add_clear_btn": _AP_ADD_CLEAR_TIP,
    "add_all_btn": "Ticks every object in the list.",
    "add_none_btn": "Unticks every object in the list.",
    "add_show_walls_cb": (
        "Shows the wall picture the cut followed, over the box."),
    # the side views only -- napari draws no outline in 3D
    "add_contour_cb": (
        "Draws the new objects as outlines in the three side views, so the "
        "membrane under them stays visible. The 3D view cannot show "
        "outlines."),
    "add_click_radius_sb": (
        "How big each click marker is: a ball of this radius, in microns."),
    "add_core_erode_sb": (
        "Each object of the other run is shrunk by this before it becomes a "
        "marker, so neighbours do not touch."),
    "add_dark_outside_cb": (
        "Dark space (no signal in any channel) stops the fill, like the box "
        "edges. Untick it if dim cells come out empty."),
    "add_dark_clearance_sb": (
        "Dark areas are left empty. The fill may reach this far into them "
        "before it stops."),
    "add_dark_factor_sb": _AP_ADD_HELP_DARK_BELOW,
    # Cells: the tissue outline and how cells grow
    "add_outline_combo": (
        "Where new cells may grow: inside the outline of the objects you "
        "cleared, inside a shrink-wrap around the stain, or (as before) "
        "wherever the box edges and dark space let them."),
    "add_show_outline_cb": (
        "Draws the edge of the tissue outline in the main view and in the "
        "three side views, so you can scroll the slices and see where the "
        "tissue ends."),
    # Nuclei: the Tissue outline card's own combo, and its Show tick (the
    # same words as Cells')
    "add_nuc_outline_combo": _AP_ADD_HELP_NUC_OUTLINE,
    "add_nuc_show_outline_cb": (
        "Draws the edge of the tissue outline in the main view and in the "
        "three side views, so you can scroll the slices and see where the "
        "tissue ends."),
    "add_grow_combo": _AP_ADD_HELP_GROW,
    "add_wall_cost_sb": _AP_ADD_HELP_WALL_COST,
    "add_growth_limit_sb": _AP_ADD_HELP_GROWTH_LIMIT,
    "add_include_membrane_cb": _AP_ADD_HELP_INCLUDE_MEMBRANE,
    "add_bridge_sb": _AP_ADD_HELP_BRIDGE,
    "add_separate_cb": (
        "Splits objects that touch, so a marker takes the nucleus it sits "
        "on, not its neighbours. Off: one object per bright blob; at the "
        "markers, two clicks in one blob still split it."),
    "add_min_distance_sb": (
        "The closest two object centres may be, in voxels. Lower gives "
        "more, smaller objects."),
    "add_marker_spacing_sb": (
        "How far apart the middles of two nuclei must be for them to count "
        "as two, in voxels of the full-size image across (the same distance "
        "in microns between slices). The default suits most nuclei; much "
        "higher joins close neighbours to the nucleus you click; much lower "
        "can let a click spread through rough, uneven stain."),
    "add_clahe_clip_sb": "How strongly Local contrast evens the box out.",
    "add_clear_border_cb": (
        "Objects that reach the box edge were probably cut by it, so they "
        "start unticked. A lone object stays ticked."),
    "add_edge_margin_sb": "How many voxels in from the box edge count as the edge.",
    "add_min_size_sb": (
        "Objects smaller than this many voxels are left out. 0 keeps every "
        "one."),
    "add_auto_smooth_cb": (
        "Smooths each new object's surface in the preview, so what you see "
        "is what Accept writes."),
    "add_smooth_sigma_sb": (
        "How much smoothing, in voxels of the full-size image. 1.5 is "
        "gentle."),
    "add_live_cb": _AP_ADD_HELP_LIVE_TICK,
    "add_live_always_cb": _AP_ADD_HELP_LIVE_ALWAYS,
    "add_batch_refine_btn": ADD_REFINE_ALL_TIP,
    "add_refine_piece_btn": (
        "Takes the object selected in the list to the Refine tab."),
    "add_again_btn": (
        "Start again in the same box, with the same preset and marker "
        "source."),
}

#: The box row's Whole / Numbers... / ⌖: PANEL_BTN_STYLE with 5 px of side
#: padding, so "S M L Whole Numbers... ⌖" fits the 290 px inside the
#: card at the 316 px viewport (B2 fix pass U-M5d).
_ADD_TIGHT_BTN_STYLE = PANEL_BTN_STYLE.replace("padding: 4px 10px;",
                                               "padding: 4px 5px;")

#: The item-data role that carries a candidate row's chips.
ADD_CHIPS_ROLE = Qt.UserRole + 1

#: What a preset switch sets, per widget attribute -- only the
#: keys the user has NOT touched this session.  The dark tick is NOT here:
#: it is not a preset value; the workspace remembers it on its own.
_ADD_PRESET_VALUES = {
    "nuclei": {
        "add_clahe_cb": False, "add_clahe_clip_sb": 0.01,
        "add_separate_cb": True, "add_min_distance_sb": 3,
        # the markers' own Spacing (Anywhere's stays 3)
        "add_marker_spacing_sb": _AP_ADD_MARKER_SPACING_DEFAULT,
        "add_clear_border_cb": True, "add_edge_margin_sb": 3,
        "add_min_size_sb": 0,
        "add_auto_smooth_cb": True, "add_smooth_sigma_sb": 1.5,
        # Inside each cell's three knobs
        "add_nuc_radius_sb": 2.5, "add_nuc_edge_sb": 0.25,
        "add_two_gap_sb": 0.70,
    },
    "cells": {
        "add_wall_kind_combo": 0,
        "add_prep_zcorrect_cb": True, "add_prep_bgsub_cb": False,
        "add_prep_lift_cb": False, "add_prep_denoise_cb": False,
        "add_click_radius_sb": 1.0, "add_core_erode_sb": 1.0,
        "add_dark_clearance_sb": 2.0, "add_dark_factor_sb": 0.5,
        "add_clear_border_cb": True, "add_edge_margin_sb": 3,
        "add_min_size_sb": 0,
        "add_auto_smooth_cb": True, "add_smooth_sigma_sb": 1.5,
        # Spread from each marker's three knobs (Grow by, Tissue outline and
        # Bridge gaps follow the box and the patch instead: see
        # ``_apply_add_cells_rules``)
        "add_wall_cost_sb": 40.0, "add_growth_limit_sb": 28.0,
        "add_include_membrane_cb": True,
    },
}
_ADD_PRESET_KEYS = tuple(sorted(
    {k for v in _ADD_PRESET_VALUES.values() for k in v}))

#: Every Add knob whose change drops a preview on screen.
_ADD_ADVANCED_SIGNALS = (
    ("add_separate_cb", "toggled"),
    ("add_min_distance_sb", "valueChanged"),
    ("add_marker_spacing_sb", "valueChanged"),
    ("add_min_size_sb", "valueChanged"),
    ("add_clahe_clip_sb", "valueChanged"),
    ("add_clear_border_cb", "toggled"),
    ("add_edge_margin_sb", "valueChanged"),
    ("add_auto_smooth_cb", "toggled"),
    ("add_smooth_sigma_sb", "valueChanged"),
    ("add_click_radius_sb", "valueChanged"),
    ("add_core_erode_sb", "valueChanged"),
    ("add_dark_clearance_sb", "valueChanged"),
    ("add_dark_factor_sb", "valueChanged"),
    ("add_dark_outside_cb", "toggled"),
    ("add_wall_kind_combo", "currentIndexChanged"),
    ("add_nuc_radius_sb", "valueChanged"),
    ("add_nuc_edge_sb", "valueChanged"),
    ("add_two_gap_sb", "valueChanged"),
    ("add_prep_zcorrect_cb", "toggled"),
    ("add_prep_bgsub_cb", "toggled"),
    ("add_prep_lift_cb", "toggled"),
    ("add_prep_denoise_cb", "toggled"),
    ("add_grow_combo", "currentIndexChanged"),
    ("add_wall_cost_sb", "valueChanged"),
    ("add_growth_limit_sb", "valueChanged"),
    ("add_include_membrane_cb", "toggled"),
    ("add_bridge_sb", "valueChanged"),
)


def wrap_and_elide(text, fm, width, max_lines):
    """``text`` word-wrapped into at most ``max_lines`` lines of ``width``
    px in ``fm``; the last line is elided with "…" when the text does not
    fit (the Add counts note, B2 fix pass U-M2).  Lines are joined with
    newlines, so a label showing it needs no word wrap of its own."""
    words = str(text or "").split()
    if not words:
        return ""
    lines = []
    cur = ""
    i = 0
    while i < len(words) and len(lines) < max_lines - 1:
        trial = (cur + " " + words[i]).strip()
        if cur and fm.horizontalAdvance(trial) > width:
            lines.append(cur)
            cur = ""
            continue
        cur = trial
        i += 1
    rest = " ".join(([cur] if cur else []) + words[i:])
    if rest:
        lines.append(fm.elidedText(rest, Qt.ElideRight, int(width)))
    return chr(10).join(lines)


def add_row_layout(text_w, name_w, chip_ws, tail_w, *, gap=6):
    """How many chips fit on one candidate row.

    The id (``name_w``) and the size (``tail_w``) are never touched; chips
    are dropped right-to-left until what is left fits.  Returns the number
    of chips drawn."""
    room = int(text_w) - int(name_w) - int(tail_w) - gap
    shown = 0
    used = 0
    for w in chip_ws:
        need = int(w) + gap
        if used + need > room:
            break
        used += need
        shown += 1
    return shown


class AddCandidateDelegate(PanelRowDelegate):
    """A candidate / REVIEW row: check box, swatch, "Object {id}", the chips
    ``edge`` / ``small`` in one line, and the size right-aligned in mono.
    Chips elide right-to-left before the id or the size are touched
    (:func:`add_row_layout`); the item TEXT stays what the tests read."""

    CHIP_PAD = 10

    def paint(self, painter, option, index):  # noqa: N802 - Qt naming
        from qtpy.QtCore import QRect
        from qtpy.QtGui import QColor, QFontMetrics
        from qtpy.QtWidgets import (
            QApplication, QStyle, QStyleOptionViewItem,
        )

        try:
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            head, sep, tail = str(opt.text or "").rpartition(self._separator)
            if not sep:
                head, tail = str(opt.text or ""), ""
            chips = list(index.data(ADD_CHIPS_ROLE) or [])
            widget = opt.widget
            style = (widget.style() if widget is not None
                     else QApplication.style())
            mono = self._mono(opt.font)
            fm = QFontMetrics(opt.font)
            fm_mono = QFontMetrics(mono)
            tail_w = (fm_mono.horizontalAdvance(tail) + 10) if tail else 0
            base = QStyleOptionViewItem(opt)
            base.text = ""
            style.drawControl(QStyle.CE_ItemViewItem, base, painter, widget)
            text_rect = style.subElementRect(QStyle.SE_ItemViewItemText,
                                             base, widget)
            name_w = fm.horizontalAdvance(head) + 8
            chip_ws = [fm.horizontalAdvance(c) + self.CHIP_PAD for c in chips]
            n = add_row_layout(text_rect.width(), name_w, chip_ws, tail_w)
            opt.text = head
            opt.rect = QRect(opt.rect.left(), opt.rect.top(),
                             max(24, opt.rect.width() - tail_w),
                             opt.rect.height())
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)
            painter.save()
            x = text_rect.left() + name_w
            for chip, w in list(zip(chips, chip_ws))[:n]:
                r = QRect(x, option.rect.top() + 3, w,
                          option.rect.height() - 6)
                painter.setPen(QColor(PANEL_GOLD))
                painter.setBrush(QColor(212, 168, 67, 50))
                painter.drawRoundedRect(r, 3, 3)
                painter.setPen(QColor("#eeeeee"))
                painter.drawText(r, Qt.AlignCenter, chip)
                x += w + 6
            if tail:
                painter.setFont(mono)
                painter.setPen(QColor(self._color))
                painter.drawText(
                    QRect(option.rect.right() - tail_w + 2, option.rect.top(),
                          tail_w - 6, option.rect.height()),
                    Qt.AlignRight | Qt.AlignVCenter, tail)
            painter.restore()
        except Exception:
            try:
                super().paint(painter, option, index)
            except Exception:
                pass


class _AddBoxNumbers(QDialog):
    """Numbers... -- the box's six sliders in a non-modal dialog
   .  Closed at every Add exit; its sliders are disabled while a
    preview runs."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("The box, in numbers")
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(2)
        # sliders only: the whole patch is a size on the box row (Whole); a
        # slider move makes a whole patch a box again
        try:
            from superqt import QLabeledSlider as _Slider
        except Exception:  # pragma: no cover - superqt is a napari dep
            from qtpy.QtWidgets import QSlider as _Slider
        self.sliders = {}
        for key, text, tip in (
            ("cz", "Centre Z", "Move the box through depth (Z)."),
            ("cy", "Centre Y", "Move the box up and down (Y)."),
            ("cx", "Centre X", "Move the box left and right (X)."),
            ("sz", "Size Z", "The box's depth (Z), in slices."),
            ("sy", "Size Y", "The box's height (Y), in voxels."),
            ("sx", "Size X", "The box's width (X), in voxels."),
        ):
            row = QHBoxLayout()
            lbl = QLabel(text, self)
            lbl.setFixedWidth(56)
            row.addWidget(lbl)
            sld = _Slider(Qt.Horizontal, self)
            sld.setRange(1, 100)
            sld.setValue(20)
            sld.setToolTip(tip)
            self.sliders[key] = sld
            row.addWidget(sld, 1)
            lay.addLayout(row)
        self.resize(320, self.sizeHint().height())

#: Orientation, read BY INDEX into `refine.CONTOUR_ORIENTATIONS`.
REFINE_CONTOUR_AXES = ("3D", "XY", "XZ", "YZ")

#: The slice sampler, read BY INDEX into `refine.SLICE_SAMPLINGS`.
REFINE_CONTOUR_SLICE_LABELS = (
    "A share of the slices",
    "A fixed number of slices",
    "Every slice (slowest)",
)

#: The methods that read an image channel, and so show the shared channel
#: card.  Every other method hides the channel row and the
#: prepare chevron entirely -- asking for a channel a method will not read
#: is the same mistake in the other direction.
REFINE_CHANNEL_METHODS = ("contour", "settle")

#: The methods that build a WALL ELEVATION, and so take the chain with
#: ``drop_clahe=True`` and may show the wall picture.
REFINE_WALL_METHODS = ("contour", "settle")

#: The Refine tab's wall kinds: the eight Reshape kinds that are not
#: "No wall", plus the reconstruction's probability map.
#:
#: "No wall (midline only)" is DROPPED rather than carried over: a flat
#: elevation makes a Voronoi partition between markers, which is meaningful
#: for a cut and meaningless for settling one outline.  The Reshape combo is
#: untouched, so ``get_split_params()`` is byte-identical.
REFINE_WALL_KINDS = tuple(k for k in WALL_KINDS if k != "none") + (
    "probability",
)

REFINE_WALLS_FROM_LABELS = tuple(
    lab for kind, lab in zip(WALL_KINDS, WALLS_FROM_LABELS) if kind != "none"
) + ("Probability map (from the reconstruction)",)

#: Which of them need an image channel.  Mirrors
#: ``reshape.spec.IMAGE_DERIVED_WALLS`` -- ``probability`` is deliberately
#: NOT one of them: it reads the reconstruction, not a channel.
REFINE_IMAGE_WALL_INDICES = frozenset(
    i for i, kind in enumerate(REFINE_WALL_KINDS)
    if kind in ("membrane", "intensity", "frangi", "meijering",
                "gradient", "hybrid")
)

#: The two SHAPE-derived kinds (B3 audit, MED).  Their picture is built from
#: the object alone -- `norm(-EDT)` for Pinch points is at its maximum
#: everywhere OUTSIDE the object -- so under Settle the default solver runs
#: the outline straight to the band (+75 % on a ball, measured).  Not
#: refused (a user may want exactly that), but said, in amber, on the page.
REFINE_SHAPE_WALL_KINDS = ("distance", "skeleton")

REFINE_SHAPE_WALL_NOTE = (
    "Shape-based walls are for splitting. For Settle, pick a channel wall "
    "type or the probability map, or with the default solver the outline "
    "will grow to the limit."
)

#: Why the probability-overlay tick came back off (B3 audit, MED).  Qt
#: shows no tooltip on a disabled widget, so the reason is written as visible
#: text under the tick, and cleared the next time the tick works.
REFINE_PROB_OVERLAY_NO_OBJECT = (
    "Select an object first - the map is shown around the selected object."
)
REFINE_PROB_OVERLAY_NO_MAP = (
    "This patch has no probability map from the reconstruction, so there "
    "is nothing to show."
)

#: The index of the probability kind, for the one row that is neither an
#: image wall nor a shape wall.
REFINE_WALLS_PROBABILITY = REFINE_WALL_KINDS.index("probability")

#: Settle's solvers -- the same three engine names as Reshape, said in the
#: words this page needs (it IS the settle tool, so "Settle onto the wall
#: signal" would name the page rather than the solver).
REFINE_SOLVERS = SPLIT_SOLVERS
REFINE_SOLVER_LABELS = (
    "Sharp edge (default)",
    "Soft edge",
    "Diffused edge (needs u-Segment3D)",
)

#: What the third solver's item says when u-Segment3D is not installed.
#: A refusal that fires AFTER the click teaches nothing,
#: so the ITEM is greyed with the reason on it -- and the reason is repeated
#: under the combo, because Qt shows no tooltip on a disabled item either.
REFINE_SOLVER_MISSING = (
    "Needs the u-Segment3D package, which is not installed - the other two "
    "work without it."
)

#: The third reason the walls tick can be grey.  Qt shows no
#: tooltip on a disabled widget, so it is written into the legend's slot,
#: exactly as its three Reshape siblings are.
WALLS_TOGGLE_NO_WALLS = "This method does not use a wall picture."

#: ...and the fourth: a chain whose LAST step has no wall picture (Smooth
#: after a Settle, B1c).  The earlier step's picture is not what this step
#: followed, so it is taken away and the tick says why.
WALLS_TOGGLE_LAST_STEP = (
    "The last step does not use a wall picture, so there is nothing to show."
)

#: Refine's own legends for the two SHAPE kinds and its own tick tooltip
#: (B3 audit): Reshape's talk about "the cut", which this tab does not make,
#: and Pinch points is bright OUTSIDE the object too -- `norm(-EDT)` of the
#: object is at its maximum wherever the EDT is 0.
REFINE_WALLS_LEGEND_PINCH = (
    "The object's own shape, not the image. Bright = outside the object "
    "and where it is thin."
)
REFINE_WALLS_LEGEND_SPINES = (
    "The object's own shape, not the image. Bright = far from the object's "
    "spine; the spine is dark."
)
REFINE_WALLS_TOGGLE_HELP = (
    "Adds the wall picture the outline followed as a layer, so you can see "
    "why it moved where it did. Remove the tick to take it away."
)

#: Preview's own tooltip.  "Nothing is FINAL", not "nothing is written": a
#: staged shape IS in the volume until Accept or Discard (B3 audit).
REFINE_PREVIEW_TIP = (
    "Run the chosen method and show the result. Nothing is final until you "
    "Accept."
)

#: ...and the tenth kind's legend sentence.
WALLS_LEGEND_PROBABILITY = (
    "Bright = where the reconstruction was least sure this was one object."
)

#: The one sentence a first-time user of this tab is guaranteed to read,
#: VERBATIM -- em dash included (B3; `test_refine_help_parity`
#: reads it out of the code).  It WRAPS beside the chevron, never elided.
REFINE_ABOUT_LINE = (
    "Fixes the shape of the object you have selected — one object at a "
    "time. No other object ever loses a voxel, and nothing is final "
    "until you press Accept."
)

#: ...and what is behind it.  Rich text, so every literal `<`, `>` or `&`
#: added here has to be escaped.
REFINE_ABOUT_BODY = (
    "Four of these never look at the image &#8212; <b>Smooth</b>, "
    "<b>Dilate/Erode</b>, <b>Fill Holes</b> and <b>Flood Fill</b> only "
    "change the shape. <b>Guided Grow</b> reads the reconstruction's "
    "probability map. <b>Active contour</b> and <b>Settle</b> read the "
    "channel you pick, unless their wall type is built from the shape or "
    "the probability map."
    "\n\n"
    "<b>Smooth</b> rounds the object off, which is what takes the "
    "slice-by-slice staircase out of it. <b>Active contour</b> slides the "
    "outline until it sits on something the image shows. "
    "<b>Dilate/Erode</b> grows or shrinks it by a fixed amount. "
    "<b>Guided Grow</b> fills outward into voxels the reconstruction "
    "thought were part of an object. <b>Fill Holes</b> closes cavities "
    "inside it. <b>Flood Fill</b> takes the empty space under the "
    "crosshair. <b>Settle onto the wall signal</b> moves the outline onto "
    "the nearest wall, and no further than you allow."
    "\n\n"
    "You can run several in a row on the same object; Discard puts it back "
    "as it was before the first step."
)


#: Disclosure chevrons.  Written as escapes so the source text stays
#: cp1252-safe wherever it is quoted.
_CHEVRON_CLOSED = "\u25b8"
_CHEVRON_OPEN = "\u25be"

#: The Reshape tab's smoothing choice, item order = combo index.
SMOOTH_MODE_LABELS = (
    "Off",
    "Smooth pieces",
    "Keep the object",
)
SMOOTH_MODE_OFF, SMOOTH_MODE_PIECES, SMOOTH_MODE_KEEP = 0, 1, 2

#: The two sigma row labels.  Same escape rule: a literal sigma in the
#: source is not cp1252-encodable, which is the class of hazard the
#: 2026-09-07 console fix exists for.
_SIGMA_Z_LABEL = "\u03c3_z:"
_SIGMA_XY_LABEL = "\u03c3_xy:"

#: The sigma pair shares one row, so it gets its own, much narrower
#: column instead of the Advanced one.
_SIGMA_LABEL_COLUMN = 30

#: The two ticklist ceilings.  A list never grows past these; below
#: them it is exactly as tall as the rows it holds (`_fit_panel_list`),
#: and the user can drag past them with the grip under it.
#:
#: 84 + the 6 px grip is the 90 px block the marker list is capped
#: at: three 26 px rows on the live dock still fit, and the handle is
#: paid for out of the same budget rather than out of the page's.
_MARKER_LIST_MAX_H = 90 - PANEL_GRIP_H
_PIECE_LIST_MAX_H = 130 - PANEL_GRIP_H

#: ...and the marker list's cap while a PREVIEW is on screen (audit H2,
#: Brick U4).  Review-with-markers-open is the tallest state the panel
#: has, and Brick U4 put one more control into it -- the "Refine
#: Fragments..." hand-off -- which took it to 778 px against a 753 px
#: ceiling.  Dropping the grey line under the marker list recovered 14
#: px; this recovers the rest.
#:
#: It is the right 11 px to take, because in review the list you ACT on
#: is the piece list: the markers are there to be CHANGED, and changing
#: one puts the preview away and gives the list its full height back.
#: A grip the user has dragged still wins -- `_fit_panel_list` reads the
#: stored property first, so this is a default, not a cap on them.
_MARKER_LIST_REVIEW_MAX_H = 73 - PANEL_GRIP_H
# `_SEED_LIST_MAX_H` went with `split_seed_list` in Brick U1: there is
# ONE marker list now and it uses `_MARKER_LIST_MAX_H`.

#: The ONE visual difference between the marker rows and the piece rows
#:.  Two identical ticklists, three slots apart, both
#: reading "<swatch> Name  ·  value", are confusable by construction --
#: and they mean opposite things: a marker decides what the cut is MADE
#: FROM, a piece decides what Accept WRITES.  The marker rows get a gold
#: left edge; the piece rows keep the plain chip.
#:
#: Appended to `PANEL_LIST_STYLE` rather than edited into `theme.py`,
#: which this round does not touch.
_MARKER_LIST_STYLE = PANEL_LIST_STYLE + (
    "QListWidget::item {"
    "  border-left: 3px solid " + PANEL_GOLD + ";"
    "}"
)

#: How far the grip may take a list, in rows.  Three is the least that
#: still reads as a list; fourteen is more than the 340x350 dock can
#: show, which is the point -- the user is telling us the dock is
#: taller than that.
_LIST_GRIP_MIN_ROWS = 3
_LIST_GRIP_MAX_ROWS = 14

#: Where a dragged height is remembered.  A Qt property on the PANEL, so
#: it survives every Find, Preview and Discard of the session and dies
#: with the dock.
_MARKER_LIST_H_PROP = "_epicycle_marker_list_h"
_PIECE_LIST_H_PROP = "_epicycle_piece_list_h"

#: A spin box must never be narrower than the number it can hold plus
#: its two buttons -- see `_make_widths_shrinkable`.
_SPIN_MIN_PAD = 40
_SPIN_MIN_W = 64

#: The separator in the collapsed S3 summary and in every list row.
#: Written as an escape for the same reason as the chevrons.
_DOT_SEP = "\u00b7"
_SUMMARY_SEP = "   " + _DOT_SEP + "   "

#: Where an elided button keeps its untruncated text, and the extra
#: sentence its tooltip appends after it.
_ELIDE_FULL_PROP = "_epicycle_full_text"
_ELIDE_HINT_PROP = "_epicycle_tooltip_hint"

#: The grey one-liner under a control -- the mockup's `.help` (theme.py).
_HELPER_LINE_STYLE = PANEL_HELP_STYLE

#: The status line beside S1's dot -- the mockup's `.status` text.
_PANEL_STATUS_TEXT_STYLE = (
    "color: #cccccc; font-size: 11px; background: transparent;"
)

#: The mockup's `.sub` -- how far a block that belongs to the row above
#: it is inset.  One number, so every sub-block lines up with every other.
_SUB_INDENT = 10

#: The mockup's `.bar` -- a 4 px gold sliver, no groove chrome.
_PANEL_BUSY_BAR_STYLE = (
    "QProgressBar {"
    "  background: rgba(255, 255, 255, 0.08);"
    "  border: none; border-radius: 2px;"
    "  max-height: 4px; min-height: 4px;"
    "}"
    "QProgressBar::chunk {"
    "  background-color: #d4a843; border-radius: 2px;"
    "}"
)

# ---------------------------------------------------------------------------
# The channel-preparation chain.
#
# The engine's `apply_preproc` splits the chain on '+' and matches it as a
# SET, then runs the steps in ONE fixed order of its own:
#     zcorrect -> bgsub -> gamma|log -> denoise -> clahe
# (preproc.py:100-120).  `chain_from_ticks` emits the tokens in that same
# order so the string a reader sees is the order that actually runs.
# ---------------------------------------------------------------------------

#: tick key -> the user's words for it.
PREP_TICK_LABELS = {
    "clahe": "Local contrast",
    "zcorrect": "Even out depth",
    "bgsub": "Remove haze",
    "lift": "Lift dim signal",
    "denoise": "Despeckle",
}

#: The order the engine runs them in, said in the same words.
_PREP_ORDER_SENTENCE = (
    "They always run in this order, whichever you tick: even out depth, "
    "remove haze, lift dim signal, despeckle, local contrast."
)


def chain_from_ticks(ticks, *, lift_token="gamma", drop_clahe=False):
    """Turn the prepare tick-boxes into an ``apply_preproc`` chain string.

    ``ticks`` is a mapping of the :data:`PREP_TICK_LABELS` keys to bools.
    Returns ``"none"`` when nothing is ticked -- the engine's own
    passthrough value, never ``""`` and never ``None``.

    ``drop_clahe`` is for the WALL chain: the wall stage applies its own
    CLAHE through the ``membrane clip`` knob, so a ``clahe`` token there
    would double-apply it.  (The engine strips it as well -- this keeps
    the string the panel SHOWS honest about what will run.)
    """
    ticks = dict(ticks or {})
    out = []
    if ticks.get("zcorrect"):
        out.append("zcorrect")
    if ticks.get("bgsub"):
        out.append("bgsub")
    if ticks.get("lift"):
        out.append("log" if str(lift_token) == "log" else "gamma")
    if ticks.get("denoise"):
        out.append("denoise")
    if ticks.get("clahe") and not drop_clahe:
        out.append("clahe")
    return "+".join(out) if out else "none"


# ---------------------------------------------------------------------------
# Helper lines.
#
# HELP_LINES is keyed by WIDGET ATTRIBUTE and rendered as a grey line
# directly under that control.  ITEM_HELP is keyed by COMBO attribute and
# indexed by item: one label under the combo whose text swaps with the
# selection.  `test_helper_lines.py` asserts set equality against an
# explicit HELPED_WIDGETS tuple, so a missing helper and an orphan entry
# both fail.
# ---------------------------------------------------------------------------

_PREP_HELP = {
    "clahe": (
        "Evens out bright and dim regions so faint nuclei count too. "
        "Usually all you need."
    ),
    "zcorrect": "Lifts deep, dim slices to match the shallow ones.",
    "bgsub": (
        "Takes away smooth background glow so only the nuclei stand out."
    ),
    "lift": "Brightens faint signal more than bright signal.",
    "denoise": (
        "Removes salt-and-pepper noise before the local-contrast step."
    ),
}

#: The one sentence a first-time user is guaranteed to read.  It
#: WRAPS beside the chevron and is never elided.
SPLIT_ABOUT_LINE = (
    "Cuts one merged object into several. Markers say where the objects "
    "are; walls say where their edges are."
)

#: ...and what is behind it.  Owner-approved text, shipped verbatim, with
#: the "object" wording throughout.
#:
#: RICH TEXT (audit M2).  The mockup emphasises the two inputs; the first
#: build had no way to and shipped "MARKERS:" / "WALLS:" in capitals,
#: which reads as shouting in a four-paragraph explainer.  The label is
#: `Qt.RichText`, so this is `<b>` and `&#8212;` rather than caps and
#: hyphens -- which also means every literal `<`, `>` or `&` added here
#: has to be escaped.
SPLIT_ABOUT_BODY = (
    "This is ordinary image processing &#8212; nothing is learned, "
    "nothing is predicted. Every step is a filter you can look at."
    "\n\n"
    "You give it two things. <b>Markers</b>: one per object you think is "
    "hiding in this blob. Paint them yourself, or let a finder look for "
    "them &#8212; a peak finder, a blob finder or a brightness "
    "threshold, in whichever channel you choose. <b>Walls</b>: a picture "
    "of where the edges are, made either from an image channel (ridges, "
    "edges, plain brightness) or from the object's own shape."
    "\n\n"
    "The cut then floods out from every marker at once and stops at the "
    "walls, so each marker usually ends up owning one piece."
    "\n\n"
    "Then it tidies up: very small pieces are folded into a neighbour, "
    "and if you said how many objects you expect, the weakest joins are "
    "put back together until there are no more than that many &#8212; it "
    "never invents a cut to reach a number. Nothing is written until you "
    "press Accept."
)

HELP_LINES = {
    # --- S2 TARGET, Brick U2 -------------------------------------------
    "split_contour_cb": (
        "Draws the object as an edge in the side views, where you paint, "
        "so you can see the image under it. napari cannot outline in 3D. "
        "No other object changes."
    ),
    # --- Markers -------------------------------------------------------
    "split_seed_prep_clahe_cb": _PREP_HELP["clahe"],
    "split_seed_prep_zcorrect_cb": _PREP_HELP["zcorrect"],
    "split_seed_prep_bgsub_cb": _PREP_HELP["bgsub"],
    "split_seed_prep_lift_cb": _PREP_HELP["lift"],
    "split_seed_prep_denoise_cb": _PREP_HELP["denoise"],
    # --- Walls ---------------------------------------------------------
    "split_wall_prep_same_cb": (
        "Uses the marker channel's preparation here too (local contrast "
        "is always applied to walls anyway)."
    ),
    "split_wall_prep_zcorrect_cb": _PREP_HELP["zcorrect"],
    "split_wall_prep_bgsub_cb": _PREP_HELP["bgsub"],
    "split_wall_prep_lift_cb": _PREP_HELP["lift"],
    "split_wall_prep_denoise_cb": _PREP_HELP["denoise"],
    "split_search_padding_sb": (
        "How far outside the object the tool looks; the value is taken "
        "when you press Start."
    ),
    # --- Brick 2b: the three rows promoted into "How to cut" ----------
    "split_refine_guided_cb": (
        "After the cut, slides each piece's edge onto the nearest edge in "
        "the wall channel. Radius and strength are under Advanced."
    ),
    "split_arbiter_cb": (
        "Merges back any cut with no wall behind it, and marks the object as "
        "dividing."
    ),
    "split_target_count_cb": (
        "If more pieces come out, the weakest joins are merged until the "
        "count matches. If fewer come out you get a warning, never a "
        "forced cut."
    ),
    # --- Advanced: MARKERS -------------------------------------------
    "split_seed_smooth_sb": (
        "How much smoothing before peaks are found. The single most useful "
        "knob: more smoothing means fewer, larger markers."
    ),
    "split_min_distance_sb": "Two markers closer together than this become one.",
    "split_seed_h_sb": (
        "Peak and interior finders: how far a peak must stand out (for "
        "interiors, how deep the basin). Blob finders: the detection "
        "threshold -- lower finds more."
    ),
    "split_max_sigma_sb": (
        "The blob finders only: blobs bigger than this are ignored."
    ),
    "split_seed_thresh_sb": (
        "Nucleus peaks and Bright-object centres: how bright a nucleus "
        "must be. On automatic the tool picks the level itself."
    ),
    "split_seed_clip_sb": (
        "Local contrast on the marker channel. Higher gives more contrast, "
        "and lets more noise through with it."
    ),
    "split_lift_mode_combo": (
        "Gentle lifts faint signal a little, strong lifts it a lot."
    ),
    "split_seed_gamma_sb": (
        "How hard the gentle lift pulls: smaller numbers lift faint signal "
        "more."
    ),
    "split_seed_bg_sb": (
        "Remove haze: the largest feature to keep, in voxels. Set it bigger "
        "than a nucleus."
    ),
    "split_seed_denoise_sb": (
        "Despeckle: how many voxels around each point are averaged. 2 is "
        "gentle."
    ),
    # --- Advanced: WALLS ---------------------------------------------
    "split_wall_clip_sb": (
        "Local contrast on the wall channel. Raise it if the wall signal is "
        "faint."
    ),
    "split_wall_smooth_sb": (
        "Smooths speckle in the wall channel before the ridge filter looks "
        "at it."
    ),
    "split_wall_ridge_cb": (
        "Turns the wall signal into a thin sheet the cut can follow. Off "
        "means use brightness directly."
    ),
    "split_arbiter_pct_sb": (
        "How strong the wall must be along a proposed cut, next to the rest "
        "of this object's wall signal. 75 % means stronger than three-quarters "
        "of it."
    ),
    # --- Advanced: CUT AND SNAP --------------------------------------
    "split_rw_beta_sb": (
        "Soft cut only. Higher makes the cut stick more tightly to walls."
    ),
    "split_compactness_sb": (
        "Sharp cut only. Above zero, pieces are pushed toward equal size -- "
        "the fix when one marker gets a starved sliver."
    ),
    "split_refine_radius_sb": "How far an edge may slide when it snaps.",
    "split_refine_eps_sb": (
        "Lower numbers snap harder to the edge in the image."
    ),
    "split_refine_merge_small_cb": (
        "Pieces smaller than the size below are folded into their biggest "
        "neighbour. A piece with its own marker is never absorbed."
    ),
    "split_min_size_sb": (
        "How small a piece has to be to count as a sliver, in voxels."
    ),
    "split_smooth_mode_combo": (
        "Rounds off the staircase edges of a hand-painted cut; the preview "
        "shows it. 'Keep the object' loses no voxel and never cuts through "
        "a marker."
    ),
    "split_auto_advance_cb": (
        "After Accept, jump straight to the next object in the triage list."
    ),
    "split_pieces_cap_sb": (
        "After Accept, the view keeps up to this many of the new pieces "
        "isolated; with more than this, only the largest piece stays."
    ),
    # --- the Detect Seeds row -------
    "split_detect_min_dist_sb": (
        "Two peaks closer together than this, in voxels, become one."
    ),
    "split_detect_bins_sb": (
        "How finely the peak heights are binned in the little histogram. "
        "More bins give finer control of the line."
    ),
    "_detect_plot_placeholder": (
        "Peaks below the line are dropped. Drag it until the marker count "
        "matches the objects you see."
    ),
    # --- REFINE, Brick B1 ------------------------------------------------
    # One line per control this brick ADDS.  The knobs B2 rebuilds keep
    # their tooltips until B2 rewrites them, so they are not listed here
    # and `HELPED_WIDGETS` does not name them.
    "refine_channel_combo": (
        "The one channel this whole tab reads. Blank means no channel is "
        "chosen: the tool says so rather than guessing one."
    ),
    # Refine's own wording (B3 audit): this tab reads a WALL or a shell
    # channel, not nuclei -- Reshape's marker ticks keep theirs.
    "refine_prep_clahe_cb": (
        "Evens out bright and dim regions so faint signal counts too. "
        "Usually all you need."
    ),
    "refine_prep_zcorrect_cb": _PREP_HELP["zcorrect"],
    "refine_prep_bgsub_cb": (
        "Takes away smooth background glow so only the signal stands out."
    ),
    "refine_prep_lift_cb": _PREP_HELP["lift"],
    "refine_prep_denoise_cb": _PREP_HELP["denoise"],
    "refine_wall_kind_combo": (
        "What counts as a wall in that channel. The two shape types and the "
        "last one read no channel."
    ),
    "refine_solver_combo": (
        "How hard the new edge is. The sharp one snaps to the strongest "
        "wall; the soft one drifts toward a faint one."
    ),
    "refine_settle_band_sb": (
        "How far the outline may move, in microns, in or out."
    ),
    "refine_settle_iter_sb": (
        "How long the diffused edge is allowed to spread. Only the "
        "diffused edge uses it; the other two ignore it."
    ),
    # B3: both said nothing about WHICH edge reads them.  `settle._diffuse`
    # is the only reader of `clamp` and `n_iter`, and `random_walker`
    # ignores `compactness` (solvers.solve) -- a knob that does nothing for
    # the chosen edge has to say so.
    "refine_settle_clamp_sb": (
        "Diffused edge only: how firmly the outline is held where it "
        "started. 1 would hold it exactly; 0 would let it dissolve."
    ),
    "refine_settle_compactness_sb": (
        "Sharp and diffused edges only: pulls the new outline toward a "
        "rounder shape. Leave it at 0 unless the result grows a spike."
    ),
    # --- REFINE, Brick B2 -- the merged Active contour.
    # The mode picker's line is PER ITEM (ITEM_HELP), so the page describes
    # only the chosen mode (trim 1).  Orientation carries no line (trim 3:
    # its explanation is the tooltip) and the push's line is LIVE, so
    # neither is here.
    "refine_contour_gate_sb": (
        "The push only acts where there is no wall. Lower this if 'Push "
        "outward' seems to do nothing; raise it to block the push in more "
        "places."
    ),
    # B3 audit: `build_elevation` reads `smooth` for membrane, hybrid and
    # gradient only -- every other kind ignores this knob.
    "refine_contour_blur_sb": (
        "Membrane sheets (plus shape) and Intensity edges only: how much the "
        "channel is blurred before walls are measured, in microns."
    ),
    "refine_contour_sharpness_sb": (
        "How hard the outline brakes on a wall. Higher stops it dead at a "
        "faint wall; lower lets it slide past one."
    ),
    "refine_contour_rebuild_um_sb": (
        "Only for a plane orientation: how smoothly the traced slices are "
        "joined back into one 3D shape, in microns."
    ),
    # --- REFINE, Brick B3 -- help parity with Reshape.  A line under every
    # knob on the five short pages, which sit far under both ceilings.  The
    # contour's own rows (Orientation, Slices/Share/Count, Edge may move,
    # Steps, the two weights) stay TOOLTIPS: the tallest state is 751 px
    # of its 763 px ceiling and Settle's 707 of 710, so a grey line there
    # is the next row that has to come off.
    "smooth_standalone_sigma_z_sb": (
        "How far the shape is blurred through depth, in voxels. Raise it "
        "to take out a slice-by-slice staircase; a thin object can blur "
        "away."
    ),
    "smooth_standalone_sigma_xy_sb": (
        "How far it is blurred within each slice, in voxels. Raise it to "
        "round off jagged edges; lower it to keep fine detail."
    ),
    # B3 audit: `smooth_label_sdf` keeps only the LARGEST piece afterwards
    # whatever the tick says -- the probe's two-piece object lost one.
    "smooth_preserve_topology_cb": (
        "If the blur would cut the object apart, a thin core holds it "
        "together. Either way only the biggest piece is kept, so an object "
        "already in pieces loses the rest."
    ),
    "refine_de_radius_sb": (
        "How many voxels to grow (+) or shrink (-). Growing only fills empty "
        "voxels, never a neighbour's; shrinking can remove a thin object."
    ),
    "refine_de_axis_combo": (
        "In 3D it works in every direction. XY, XZ or YZ works within that "
        "plane only; each slice is changed on its own."
    ),
    "refine_gg_max_radius_sb": (
        "How many voxels it may grow out from the outline, one step at a "
        "time. At most 15, the margin of the region the tool looks at."
    ),
    "refine_gg_prob_threshold_sb": (
        "How sure the reconstruction must have been, from 0 to 1 across the "
        "range around this object. Lower takes more voxels."
    ),
    "refine_gg_axis_combo": (
        "In 3D it grows in every direction. XY, XZ or YZ grows within that "
        "plane only; each slice is changed on its own."
    ),
    "refine_fh_axis_combo": (
        "In 3D only a cavity sealed on every side is filled. A plane fills "
        "any hole sealed within one slice, even one open above or below."
    ),
    "refine_ff_axis_combo": (
        "In 3D the whole gap at the crosshair is filled. XY, XZ or YZ fills "
        "only the part of it on the crosshair's slice."
    ),
    # B3 audit HIGH: the size limit is the SECOND check now.
    "refine_ff_max_size_sb": (
        "A safety limit in voxels. The gap must also touch this object and "
        "be closed in; open background is refused at any size."
    ),
    # "every view" is exact: `_toggle_prob_overlay` adds it to the 3D view
    # AND the three side views.
    "refine_prob_overlay_cb": (
        "Shows the reconstruction's confidence around the selected object, "
        "in the 3D view and the side views; brighter is more sure."
    ),
}

ITEM_HELP = {
    # REFINE B3 -- rendered as per-ITEM TOOLTIPS in the open list, not as a
    # swapping line (`AdvancedFeaturesWidget._apply_item_tooltips`): they
    # help the user CHOOSE, while the list is open, which is exactly what a
    # line under the combo cannot do -- and a second line would push Settle
    # (707 of 710 px) and the contour (751 of 763 px) over their ceilings.
    # The method's own long description is still the line under the combo.
    "refine_method_combo": (
        "Rounds the object off and takes out the slice-by-slice staircase. "
        "Reads no image.",
        "Slides the outline onto what the chosen channel shows, in 3D or "
        "slice by slice.",
        "Grows or shrinks the object by a fixed number of voxels. Reads no "
        "image.",
        "Grows into empty voxels the reconstruction thought were part of an "
        "object. Needs its probability map.",
        "Closes cavities inside the object. Reads no image.",
        "Gives this object the empty space under the crosshair. Reads no "
        "image.",
        "Moves the outline onto the nearest wall, no further than you allow. "
        "No other object loses a voxel.",
    ),
    # Refine's nine wall kinds (REFINE_WALL_KINDS order).  What the wall
    # PICTURE is, read off `reshape/walls.py`, in words for an outline --
    # Reshape's lines talk about a cut, which this tab does not make.
    "refine_wall_kind_combo": (
        "Not the image: the object's own shape, bright where it is thin. "
        "Meant for splitting; under Settle the outline grows to the limit.",
        "Local contrast, blur, then a sheet-ridge filter.",
        "Mostly the membrane sheets, with the object's shape as a "
        "tie-breaker.",
        "Bright where the brightness changes fastest - the edge between a "
        "bright and a dark region.",
        "The channel after local contrast, with no ridge filter.",
        "Local contrast, then a tube-shaped ridge filter, for rounded "
        "walls.",
        "A ridge filter for weak, thin walls.",
        "Not the image: bright far from the object's spine. Meant for "
        "splitting; under Settle the outline can run to the limit.",
        "Not a channel: the reconstruction's own map, bright where it was "
        "least sure this was one object.",
    ),
    "refine_solver_combo": (
        "Floods out from the object, its neighbours and the background at "
        "once, and puts the edge on the highest wall between them.",
        "Walks out from the same places, so the edge eases onto a faint wall "
        "instead of snapping to the nearest strong one.",
        "Starts from the sharp edge, then lets it spread until it settles on "
        "the wall signal. Needs u-Segment3D.",
    ),
    # REFINE B2, trim 1: the mode's line describes the CHOSEN mode only.
    # ...and the slice sampler's line describes the CHOSEN sampling (UX-L).
    "refine_contour_slices_combo": (
        "A share of the slices, spread evenly, first and last always in. "
        "The rest is filled in the way interpolation fills it.",
        "This many slices, spread evenly, first and last always in. The "
        "rest is filled in the way interpolation fills it.",
        "Every slice the object is on, up to 60 - the slowest, and the "
        "most faithful.",
    ),
    "refine_contour_mode_combo": (
        "Follows the wall signal: the outline moves until it sits on a "
        "visible edge.",
        "Needs no visible edge: compares a band just inside the outline "
        "with a band just outside.",
    ),
    "split_seeds_from_combo": (
        "Your painted markers, one per object.",
        "Peaks of the object's own thickness. Geometry-blind, so it can "
        "shatter columnar objects - a last resort.",
        "Each instance of a trusted label layer that falls inside this "
        "object.",
        "The same, read from a mask file on disk.",
        "Looks in an image channel for one marker per object.",
    ),
    "split_seed_detector_combo": (
        "Blurs the nuclei channel and puts a marker at each local peak.",
        "Only peaks that rise a set height above their surroundings.",
        "Blob centres. Touching nuclei still come out separate.",
        "The same idea, cheaper.",
        "Thresholds the channel, then puts one marker at the centre of "
        "each bright object.",
        "Each dim basin fully enclosed by membrane; the object's own edge "
        "counts as a wall.",
    ),
    "split_walls_from_combo": (
        "The narrowest neck of the object's own shape - what this tool has "
        "always done.",
        "Local contrast, blur, then a sheet-ridge filter.",
        "Mostly the membrane sheets, with the object's shape as a "
        "tie-breaker.",
        "Cuts where brightness changes fastest.",
        "The channel as it is, with no ridge filter.",
        "A tube-shaped ridge filter, for rounded walls.",
        "A ridge filter for weak, thin walls.",
        "Cuts midway between the shape's internal spines.",
        "No wall at all - cuts halfway between the markers.",
    ),
    "split_solver_combo": (
        "Floods out from each marker and cuts on the highest ridge "
        "between them.",
        "Walks probabilities out from the markers, so the cut eases onto "
        "a faint wall instead of snapping to the nearest strong one.",
        "Starts from the sharp cut, then lets the labels spread until the "
        "boundary settles on the wall signal. Needs u-Segment3D.",
    ),
}

# ---------------------------------------------------------------------------
# The curation tools panel skeleton.
#
# Every visible tab is one scroll area (S1 STATUS / S2 TARGET / S3 SOURCES /
# S4 PREVIEW / S5 RESULT / ADVANCED) plus a fixed two-button FOOTER, both
# children of the tab page.
# ---------------------------------------------------------------------------

#: Two ceilings, not one: a single global number passes happily
#: while the footer has eaten the viewport.  PROVISIONAL until the dev-PC
#: measurement -- both are edited once, together, after it.
#:
#: Measured offscreen on this branch, every tab x state (idle / seeding /
#: solving / 3-piece review / SOURCES+ADVANCED open / Add box+candidates):
#: panel minimum 229 px CONSTANT, footer 47 px CONSTANT.  Constant is the
#: point -- before the rework the panel's minimum grew with whatever the
#: tallest page happened to be showing and never shrank back.  The
#: headroom below is for the dev PC's larger fonts.
_MAX_PANEL_H = 400        # widget.minimumSizeHint().height(), any tab x state
_MAX_FOOTER_H = 90        # footer.sizeHint().height(), any state

#: S1's state dot.  A styled 8px disc rather than a glyph, so no new
#: user-visible string enters the file and the colour can carry
#: the state on its own.
_STATUS_DOT_BASE = "border-radius: 4px; background-color: {};"
#: The shared header's description chevron button.  It was a fixed 22 px,
#: and the themed chevron needs 26 -- live, the "▸" drew as a 1 px sliver
#: on all three pages (Add live-shot fix V1).  Sized to its own hint, never
#: under the 26 px measured live.
ABOUT_BTN_MIN_WIDTH = 26


def _about_btn_width(btn) -> int:
    try:
        hint = int(btn.sizeHint().width())
    except Exception:
        hint = 0
    return max(ABOUT_BTN_MIN_WIDTH, hint)


_STATUS_DOT_STYLES = {
    "idle": _STATUS_DOT_BASE.format("#777777"),
    "ready": _STATUS_DOT_BASE.format("#6abf69"),
    "busy": _STATUS_DOT_BASE.format("#e0a030"),
    "review": _STATUS_DOT_BASE.format("#5ba4cf"),
    # A refusal: the page did not do what was asked
    # (amber S1; B1b audit UX-L3).  Same amber as "busy", named
    # for what it MEANS so a refusal is not written as "busy".
    "warn": _STATUS_DOT_BASE.format("#e0a030"),
}


class _WheelGuard(QObject):
    """A wheel over an UNFOCUSED input scrolls the page, never the value.

    Everything on this panel now lives inside a scroll area, so
    a spin box under the pointer would otherwise swallow the gesture and
    silently change a segmentation parameter the user was scrolling past.
    Paired with ``Qt.StrongFocus`` on every input, so scrolling cannot give
    an input the focus that would legitimately let it take the wheel.
    """

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        try:
            if event.type() != QEvent.Wheel:
                return False
            if obj.hasFocus():
                return False
            area = obj.parentWidget()
            while area is not None and not isinstance(area, QScrollArea):
                area = area.parentWidget()
            if area is None:
                # No page to scroll -- mode="hitl_2d" has no scroll area at
                # all.  Eating the wheel there would take the gesture away
                # from the control and give it to nothing.
                return False
            QApplication.sendEvent(area.viewport(), event)
            return True
        except Exception:
            return False

# ---------------------------------------------------------------------------
# SAM variant registry — single source of truth for available models.
# ---------------------------------------------------------------------------
SAM_VARIANTS: dict[str, dict] = {
    "vit_h": {"label": "SAM ViT-H (default)", "family": "sam"},
    "vit_l": {"label": "SAM ViT-L", "family": "sam"},
    "vit_b": {"label": "SAM ViT-B (light)", "family": "sam"},
    # Future variants — uncomment when backend adapters are ready:
    # "sam2_hiera_l": {"label": "SAM 2 Hiera-L", "family": "sam2"},
    # "sam2_hiera_b": {"label": "SAM 2 Hiera-B+", "family": "sam2"},
    # "medsam": {"label": "MedSAM", "family": "medsam"},
    # "efficient_sam_s": {"label": "EfficientSAM-S", "family": "efficient_sam"},
}


def _apply_elided_text(button) -> None:
    """Re-elide a button's stored full text to its CURRENT width.

    The collapsed S3 summary names both choices and is longer than the
    340 px dock at almost every setting.  Clipping it would silently
    truncate the second half; eliding says so with an ellipsis and the
    tooltip carries the whole line.
    """
    try:
        full = str(button.property(_ELIDE_FULL_PROP) or "")
        if not full:
            return
        fm = button.fontMetrics()
        avail = max(40, int(button.width()) - 24)
        button.setText(fm.elidedText(full, Qt.ElideRight, avail))
    except Exception:
        pass


class _ListHeightGrip(QFrame):
    """The drag handle under a ticklist (owner request, 2026-09-18).

    A list that hugs its rows is right until there are twenty markers,
    at which point "as tall as three rows" is the panel deciding how
    much of the user's screen their data may have.  So: drag to resize
    between three and fourteen rows, double-click to go back to hugging.

    It mirrors the list's own visibility rather than being shown and
    hidden by every call site -- the list is made visible in five
    places, and a handle under a hidden list is a handle to nothing.
    """

    def __init__(self, panel, list_attr: str, prop: str, cap: int,
                 parent=None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._list_attr = str(list_attr)
        self._prop = str(prop)
        self._cap = int(cap)
        self._press_y = None
        self._press_h = None
        self.setFixedHeight(PANEL_GRIP_H)
        self.setStyleSheet(PANEL_GRIP_STYLE)
        self.setCursor(Qt.SizeVerCursor)
        self.setToolTip(
            "Drag to make this list taller or shorter. "
            "Double-click to fit it to its rows again."
        )
        lst = self._target()
        if lst is not None:
            lst.installEventFilter(self)
            self.setVisible(lst.isVisible())

    # -- the list it belongs to ------------------------------------
    def _target(self):
        return getattr(self._panel, self._list_attr, None)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        try:
            if obj is self._target():
                if event.type() == QEvent.Show:
                    self.setVisible(True)
                elif event.type() == QEvent.Hide:
                    self.setVisible(False)
        except Exception:
            pass
        return False

    # -- the drag ---------------------------------------------------
    @staticmethod
    def _event_y(event) -> int:
        """Qt5 and Qt6 name the global position differently."""
        try:
            return int(event.globalPosition().y())
        except Exception:
            try:
                return int(event.globalPos().y())
            except Exception:
                return 0

    def drag_by(self, delta: int) -> None:
        """Resize by *delta* px from where the drag started.

        Public so a test can drive it without a synthetic mouse: the
        handler is the behaviour, the mouse is Qt's business.
        """
        lst = self._target()
        if lst is None:
            return
        base = self._press_h
        if base is None:
            base = int(lst.height())
        self._panel._set_list_height(
            self._list_attr, self._prop, base + int(delta))

    def reset(self) -> None:
        self._panel._reset_list_height(
            self._list_attr, self._prop, self._cap)

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        lst = self._target()
        self._press_y = self._event_y(event)
        self._press_h = int(lst.height()) if lst is not None else None

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt naming
        if self._press_y is None:
            return
        self.drag_by(self._event_y(event) - self._press_y)

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt naming
        self._press_y = None
        self._press_h = None

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt naming
        self._press_y = None
        self._press_h = None
        self.reset()

    # -- three dots, painted ---------------------------------------
    def paintEvent(self, event):  # noqa: N802 - Qt naming
        super().paintEvent(event)
        try:
            from qtpy.QtGui import QColor, QPainter

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#999999"))
            cx = self.width() // 2
            cy = self.height() // 2
            for dx in (-5, 0, 5):
                painter.drawEllipse(cx + dx - 1, cy - 1, 2, 2)
            painter.end()
        except Exception:
            pass


class _ElideGuard(QObject):
    """Keeps an elided button truthful as the dock is resized.

    One filter object for the whole panel; it only ever re-runs the
    elision, so it costs one font-metrics call per resize event.
    """

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        try:
            if event.type() == QEvent.Resize:
                _apply_elided_text(obj)
        except Exception:
            pass
        return False


class _ClickableLabel(QLabel):
    """A wrapping QLabel that forwards a click.

    The visible half of the page description.  It is a
    label and not a button because a `QPushButton` cannot word-wrap and
    asks the layout for the full width of its text -- 759 px for this
    sentence, in a 316 px viewport with the horizontal scrollbar off, so
    it would be silently clipped.  Forwarding the click keeps the
    chevron and the sentence one control.
    """

    #: On top of `PANEL_HELP_STYLE`: the label is a link in everything
    #: but name, and without a hover cue nothing says so (audit M3).
    HOVER_STYLE = (
        "QLabel:hover { color: #dddddd; text-decoration: underline; }"
    )

    def __init__(self, on_click, parent=None):
        super().__init__(parent)
        self._epicycle_on_click = on_click
        try:
            self.setCursor(Qt.PointingHandCursor)
        except Exception:
            pass

    def mousePressEvent(self, event):  # noqa: N802 - Qt
        # LEFT only (audit M3).  A right-click belongs to the context
        # menu; opening a disclosure under it would be the label
        # answering a question it was not asked.
        left = True
        try:
            left = event.button() == Qt.LeftButton
        except Exception:
            left = True
        if left:
            try:
                self._epicycle_on_click()
            except Exception:
                pass
        super().mousePressEvent(event)


class _CurrentPageStack(QWidget):
    """A page stack that is as tall as the page it SHOWS.

    NOT a `QStackedWidget`, on purpose.  A stacked layout sizes itself to
    its TALLEST page, and a parent layout asks the widget's LAYOUT for its
    height-for-width (`QWidgetItem` bypasses the widget's own override when
    it has a layout) -- so every Refine method paid for the tallest page's
    rows, and after Brick B2's reorder the contour page was laid out at the
    Smooth page's 201 px with 47 px of rows in it.  Here every page sits in
    one plain box and the ones not showing are HIDDEN, which a box layout
    leaves out entirely.

    The slice of the `QStackedWidget` API the panel uses: `addWidget`,
    `setCurrentIndex`, `currentIndex`, `currentWidget`, `widget`,
    `count`, `indexOf`, and a `currentChanged(int)` signal.
    """

    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._pages = []
        self._current = -1

    def addWidget(self, page):  # noqa: N802 - the QStackedWidget name
        self._pages.append(page)
        self._lay.addWidget(page)
        if self._current < 0:
            self._current = 0
            page.setVisible(True)
        else:
            page.setVisible(False)
        return len(self._pages) - 1

    def count(self):
        return len(self._pages)

    def widget(self, index):
        if 0 <= int(index) < len(self._pages):
            return self._pages[int(index)]
        return None

    def indexOf(self, page):  # noqa: N802
        try:
            return self._pages.index(page)
        except ValueError:
            return -1

    def currentIndex(self):  # noqa: N802
        return self._current

    def currentWidget(self):  # noqa: N802
        return self.widget(self._current)

    def setCurrentIndex(self, index):  # noqa: N802
        index = int(index)
        if not 0 <= index < len(self._pages) or index == self._current:
            return
        for i, page in enumerate(self._pages):
            page.setVisible(i == index)
        self._current = index
        self.updateGeometry()
        self.currentChanged.emit(index)


class AdvancedFeaturesWidget(QWidget):
    """Dockable widget housing segmentation workflows.

    Parameters
    ----------
    mode : str
        ``"3d_curation"`` — full napari-multiview-annotator workflows.
        ``"hitl_2d"`` — lightweight HITL-specific tools only.
    """

    def __init__(self, parent: QWidget | None = None, *, mode: str = "3d_curation") -> None:
        super().__init__(parent)
        self._mode = mode
        #: tab name -> {page, scroll, content, footer, status, preview}.
        #: One entry per page built on the S1-S6 skeleton; the generic
        #: stage helpers below read it instead of hard-coding widget names.
        self._panel_slots: dict = {}
        self._wheel_guard = _WheelGuard(self)
        self._elide_guard = _ElideGuard(self)
        #: combo attribute -> the QLabel under it that swaps with the
        #: selection (ITEM_HELP).
        self._item_help_labels: dict = {}
        #: Re-entrancy guard for the one-open-at-a-time sub-sections.
        self._split_subsection_syncing = False
        #: S3 is collapsed (not unchecked) while a solve or a review owns
        #: the panel; S5 is shown only while it has something to say.
        self._split_sources_locked = False
        self._split_result_shown = False
        #: The three flags S1's dot and the S3/S5 swaps are derived from.
        self._split_is_busy = False
        self._split_is_active = False
        self._split_review_active = False
        #: ...and the fourth.  Fragment refine has
        #: written the pieces and REMOVED the Refine layer, so every
        #: control that acts on markers is acting on something that is
        #: no longer there.  It takes the same lock a running solve does.
        self._split_fragment_active = False
        #: Why Accept is grey, for the footer's one line (fix pass H3).
        self._split_footer_reason = ""
        #: What the marker-work block looked like before a solve took it
        #: away, so releasing the lock restores it rather than lighting
        #: everything up (fix pass M1).  None means "not locked".
        self._marker_work_was_enabled = None
        #: One delegate for every ticklist on the panel (theme.py).
        self._panel_row_delegate = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API — runtime registration of new SAM variants
    # ------------------------------------------------------------------

    @classmethod
    def register_sam_variant(cls, key: str, config: dict) -> None:
        """Register a new SAM variant at runtime.

        Parameters
        ----------
        key : str
            Short identifier used as combo-box data (e.g. ``"sam2_hiera_l"``).
        config : dict
            Must contain ``"label"`` (display name) and ``"family"`` (backend
            family string).
        """
        SAM_VARIANTS[key] = config

    # ------------------------------------------------------------------
    # Public API -- tab focus BY NAME
    # ------------------------------------------------------------------

    def focus_tab(self, name: str) -> bool:
        """Bring the tab titled *name* to the front.  True when it existed.

        The orchestrator used to hard-code ``setCurrentIndex(5)`` for the
        Refine hand-off, which pins a construction order the panel rework
        is about to relabel.  Name lookup is index-free, so renaming a tab
        cannot silently retarget the hand-off at a different tool.

        Never raises: every consumer reaches this widget through a
        ``getattr``-guarded dock lookup, so this must not be the thing that
        breaks a curation flow.
        """
        tabs = getattr(self, "_tool_tabs", None)
        if tabs is None:
            return False
        try:
            for i in range(tabs.count()):
                if tabs.tabText(i) == name:
                    tabs.setCurrentIndex(i)
                    return True
        except Exception:
            return False
        return False

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        if self._mode == "hitl_2d":
            self._build_hitl_2d_sections(root)
        else:
            self._build_3d_curation_sections(root)

        # Apply themed stylesheets to all inputs and inline description
        # labels in a single sweep — matches the Label Navigator + 3D
        # Curation Tools dock visual language.  Idempotent — only
        # applies to widgets that don't already have a stylesheet set
        # (so the Preview/Accept/Reject button colors from the
        # refinement-styling pass are preserved).
        self._apply_themed_styles()
        # Everything now lives inside a scroll area, so a wheel over an
        # input must scroll the page, not edit a parameter.
        self._install_wheel_guards()
        # ... and nothing may demand more width than the dock has
        # (the horizontal scrollbar is off, so overflow is clipping).
        self._make_widths_shrinkable()

    # ------------------------------------------------------------------
    # The S1-S6 page skeleton
    # ------------------------------------------------------------------

    def _page_with_footer(self):
        """A tab page = one scroll area + one fixed footer, both its children.

        The scroll area goes INSIDE the page rather than at the widget root
        for two reasons: every footer is a page widget, so a
        shared footer outside the tab widget would have no owner; and napari
        injects a vertical stretch into our root layout unless the child
        it finds is vertically Expanding, which is why both the tab strip
        and each scroll area declare that policy.

        Returns ``(page, content_layout, footer, scroll)`` -- build into
        ``content_layout`` for S1-S5 + ADVANCED, and into ``footer`` for S6.
        """
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(2)

        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 2, 4, 2)
        # The cards carry the rhythm now (mockup `.sec`), so the gaps
        # BETWEEN them pay for the padding INSIDE them: 3 px -> 2 px.
        content_layout.setSpacing(2)
        scroll.setWidget(content)

        page_layout.addWidget(scroll, 1)

        footer = QWidget(page)
        # `.foot` -- a hairline and a slightly raised surface, so the two
        # buttons that act on the cell read as pinned rather than as the
        # last row of the page.
        footer.setObjectName(PANEL_FOOTER_OBJECT)
        footer.setAttribute(Qt.WA_StyledBackground, True)
        footer.setStyleSheet(PANEL_FOOTER_STYLE)
        page_layout.addWidget(footer, 0)

        return page, content_layout, footer, scroll

    def _register_page(self, name, page, scroll, content, footer,
                       *, status=None, preview=None) -> None:
        """Record a page's slots so the generic stage helpers can find them."""
        self._panel_slots[name] = {
            "page": page, "scroll": scroll, "content": content,
            "footer": footer, "status": status, "preview": preview,
        }

    def _footer_summary_label(self) -> QLabel:
        """The footer's one line -- reserved at a FIXED two lines.

        It wraps to two lines at the dock's 340 px for the longer states
        and to one for the shorter ones, which moved the footer (and with
        it the panel's minimum height) every time the stage changed: the
        live driver measured 54 px -> 69 px across states.  The footer is
        the one thing on this panel that must not move, so the slot is
        reserved at its worst case and the text sits at the top of it.

        The height comes from the label's OWN font metrics, so it scales
        with the machine's font rather than assuming this one.
        """
        label = QLabel("")
        label.setWordWrap(True)
        label.setStyleSheet(PANEL_HELP_STYLE)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # Two lines plus the inter-line leading Qt adds, plus a little
        # slack: over-reserving a few pixels is invisible, clipping the
        # second line is not.
        _fm = label.fontMetrics()
        label.setFixedHeight(_fm.height() * 2 + _fm.leading() + 6)
        return label

    def _status_row(self, dot_attr: str, label: QWidget) -> QWidget:
        """S1: a state dot plus the tool's status line, on one row.

        The mockup's `.status` -- a pill, not a bare row, so the one line
        that says what the tool is doing is separated from the sections
        that say what it will do next.
        """
        row = QWidget()
        row.setObjectName(PANEL_STATUS_PILL_OBJECT)
        row.setAttribute(Qt.WA_StyledBackground, True)
        row.setStyleSheet(PANEL_STATUS_PILL_STYLE)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(7, 3, 7, 3)
        lay.setSpacing(6)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(_STATUS_DOT_STYLES["idle"])
        setattr(self, dot_attr, dot)
        lay.addWidget(dot, 0, Qt.AlignVCenter)
        lay.addWidget(label, 1)
        return row

    def _set_add_s1_kind(self, kind: str) -> None:
        """The Add S1 dot AND its text colour, from one kind:
        amber text for "warn", the dim grey otherwise."""
        self._set_status_dot("_add_status_dot", kind)
        lbl = getattr(self, "add_status_label", None)
        if lbl is not None:
            lbl.setStyleSheet(ADD_S1_WARN_STYLE if kind == "warn"
                              else ADD_S1_STYLE)

    def _set_status_dot(self, dot_attr: str, state: str) -> None:
        dot = getattr(self, dot_attr, None)
        if dot is None:
            return
        try:
            dot.setStyleSheet(
                _STATUS_DOT_STYLES.get(state, _STATUS_DOT_STYLES["idle"])
            )
        except Exception:
            pass

    def _focus_panel_stage(self, name: str, *, reveal=None) -> None:
        """After a stage swap: the thing to decide in view, focus on Preview.

        Collapsing SOURCES and inserting RESULT changes the
        content height under a retained scroll position, and Qt reassigns
        focus when a focused widget is hidden -- with Accept as the likely
        default button, a held Return could commit.

        ``reveal`` names the widget the swap just produced.  At the dock's
        340x350 the RESULT starts below the fold, so scrolling to the top
        would put the user's eyes on the status line while the pieces they
        have to tick are off screen; every other swap still goes back to S1.
        """
        slots = self._panel_slots.get(name)
        if not slots:
            return
        try:
            scroll = slots.get("scroll")
            if scroll is not None:
                if reveal is not None:
                    scroll.ensureWidgetVisible(reveal)
                else:
                    bar = scroll.verticalScrollBar()
                    if bar is not None:
                        bar.setValue(bar.minimum())
            preview = slots.get("preview")
            if preview is not None and preview.isEnabled():
                preview.setFocus(Qt.OtherFocusReason)
        except Exception:
            pass

    def _make_widths_shrinkable(self) -> None:
        """No control may force the content wider than the dock.

        The horizontal scrollbar is OFF, so anything whose
        minimum width exceeds the 340 px the launcher opens is silently
        CLIPPED -- not scrolled to, not elided, just gone.  Two classes
        do it: a combo, whose minimum is its longest ITEM ("nuclei
        labels from another run of this patch" is 43 characters), and an
        elided button, whose minimum is its own text -- which is a
        feedback loop, because the text is elided to that same width and
        so never shrinks.

        Both are cut by making the horizontal policy ``Ignored``: the
        widget takes the width it is given instead of demanding one.  The
        combo's POPUP keeps a generous minimum, so the full item text is
        still readable where it matters.
        """
        from qtpy.QtWidgets import (
            QComboBox as _QComboBox,
            QDoubleSpinBox as _QDoubleSpinBox,
            QSpinBox as _QSpinBox,
        )

        for combo in self.findChildren(_QComboBox):
            try:
                combo.setSizePolicy(
                    QSizePolicy.Ignored,
                    combo.sizePolicy().verticalPolicy(),
                )
                combo.setMinimumContentsLength(6)
                view = combo.view()
                if view is not None:
                    view.setMinimumWidth(320)
            except Exception:
                pass
        # A spin box's minimum comes from its range and its decimals --
        # `split_rw_beta_sb` alone asks for 1000.000 -- and the Advanced
        # form stacks a column of them beside a column of labels.
        #
        # But `Ignored` means a minimum of ZERO, and a layout will take
        # that literally: on the Detect row it left "Min dist" as a bare
        # minus button with no number in it (owner, 2026-09-18).  So the
        # sizeHint is ignored and an explicit, READABLE minimum is put
        # back -- the widest value it can hold, plus its two buttons.
        # An explicit minimumSize outranks the policy in qSmartMinSize,
        # which is exactly the escape hatch this needs.
        for cls in (_QSpinBox, _QDoubleSpinBox):
            for box in self.findChildren(cls):
                try:
                    box.setSizePolicy(
                        QSizePolicy.Ignored,
                        box.sizePolicy().verticalPolicy(),
                    )
                    # a box whose range is far wider than its values
                    # carries its own readable floor (Add's nuclei
                    # threshold: a 1e6 range, 16-bit values)
                    floor = box.property("_epicycle_readable_floor")
                    box.setMinimumWidth(
                        int(floor) if floor else
                        self._readable_spin_width(box))
                except Exception:
                    pass
        self._invalidate_layout_cache()

    def _invalidate_layout_cache(self) -> None:
        """Throw away every cached layout minimum, once, after the sweep.

        Qt caches a layout's minimum and only recomputes it when the
        layout is invalidated.  `setSizePolicy` invalidates the widget's
        OWN parent layout, but a minimum that has already been rolled up
        into a grandparent's cache stays there -- so the sweep above
        made every combo shrinkable and the page went on asking for the
        widest combo's sizeHint anyway.

        Measured on the Refine page (Brick B1): the content asked for
        **478 px in a 316 px viewport** -- and the horizontal scrollbar
        is off, so that overflow is CLIPPED, not scrolled to.  After one
        recursive invalidate the same page asks for **257 px**.  The
        Reshape numbers move the same way or not at all; nothing here can
        make a minimum LARGER.
        """
        seen = set()

        def walk(lay):
            if lay is None or id(lay) in seen:
                return
            seen.add(id(lay))
            try:
                lay.invalidate()
            except Exception:
                return
            for i in range(lay.count()):
                item = lay.itemAt(i)
                if item is None:
                    continue
                sub = item.layout()
                if sub is not None:
                    walk(sub)
                    continue
                w = item.widget()
                if w is not None:
                    walk(w.layout())

        # ONE walk from the root, plus one per PAGE.  The first build did
        # `for child in self.findChildren(QWidget): walk(child.layout())`,
        # which re-walks every subtree once per widget in it -- quadratic,
        # and measured at >25 MINUTES for a test file that builds the panel
        # a few hundred times.  The root walk alone is not enough either:
        # a `QTabWidget` keeps its pages in a private stack that
        # `layout()` does not expose, so the walk stops at the tab strip.
        # The registered page slots are exactly the way back in, and there
        # are six of them.
        walk(self.layout())
        for slots in (getattr(self, "_panel_slots", None) or {}).values():
            for key in ("page", "content", "footer"):
                w = slots.get(key)
                if w is not None:
                    walk(w.layout())

    @staticmethod
    def _readable_spin_width(box) -> int:
        """The px a spin box needs to SHOW its largest value.

        Its own font metrics, so it scales with the machine's font, and
        the widest of the plain and the formatted rendering of the
        maximum -- a QDoubleSpinBox prints 5.0 as "5.00".
        """
        try:
            fm = box.fontMetrics()
            texts = [str(box.maximum()), str(box.minimum())]
            # A box whose minimum shows a WORD ("automatic") instead of a
            # number is only readable if the word fits.
            special = str(box.specialValueText() or "")
            if special:
                texts.append(special)
            try:
                texts.append("{:.{}f}".format(
                    float(box.maximum()), int(box.decimals())))
            except Exception:
                pass
            widest = max(fm.horizontalAdvance(t) for t in texts)
            extra = fm.horizontalAdvance(
                str(box.prefix() or "") + str(box.suffix() or ""))
            return max(_SPIN_MIN_W, widest + extra + _SPIN_MIN_PAD)
        except Exception:
            return _SPIN_MIN_W

    def _install_wheel_guards(self) -> None:
        """StrongFocus + the wheel filter on every input."""
        from qtpy.QtWidgets import (
            QComboBox as _QComboBox,
            QDoubleSpinBox as _QDoubleSpinBox,
            QSpinBox as _QSpinBox,
        )

        guard = getattr(self, "_wheel_guard", None)
        if guard is None:
            return
        for cls in (_QComboBox, _QSpinBox, _QDoubleSpinBox):
            for w in self.findChildren(cls):
                try:
                    w.setFocusPolicy(Qt.StrongFocus)
                    w.installEventFilter(guard)
                except Exception:
                    pass

    def _build_hitl_2d_sections(self, root: QVBoxLayout) -> None:
        """Build HITL 2D advanced features — SAM, classic 2D tools."""
        self._tool_tabs = QTabWidget()
        self._tool_tabs.setStyleSheet(TAB_WIDGET_STYLE)
        self._tool_tabs.addTab(self._build_sam_section(), "SAM")
        self._tool_tabs.addTab(self._build_otsu_section(), "Otsu")
        self._tool_tabs.addTab(self._build_watershed_section(), "Watershed")
        root.addWidget(self._tool_tabs)

    def _build_3d_curation_sections(self, root: QVBoxLayout) -> None:
        """Build full napari-multiview-annotator workflows for 3D curation."""
        self._tool_tabs = QTabWidget()
        self._tool_tabs.setStyleSheet(TAB_WIDGET_STYLE)
        # napari's _maybe_add_vertical_stretch walks the root layout's direct
        # children and appends a stretch unless one is vertically Expanding
        # (qt_viewer_dock_widget.py).  QTabWidget defaults to Preferred, so
        # without this the injected stretch fights the pages from outside.
        self._tool_tabs.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )
        self._tool_tabs.addTab(self._build_sam_section(), "SAM")
        self._tool_tabs.addTab(self._build_membrane_walker_section(), "Membrane")
        self._tool_tabs.addTab(self._build_scribble_section(), "Scribble")
        self._tool_tabs.addTab(self._build_histogram_section(), "Histogram")
        # "Reshape" is the Split tab's ship name.  The tab
        # TEXT moved; the construction ORDER did not, so Refine keeps index 5
        # and the four stubs keep 0-3.
        self._tool_tabs.addTab(self._build_assisted_split_section(), "Reshape")
        self._tool_tabs.addTab(self._build_refine_section(), "Refine")
        # "Add" (de-novo segmentation) is appended LAST so Refine keeps its
        # hard-referenced index 5 (the pieces→Refine handoff relies on it).
        self._tool_tabs.addTab(self._build_add_denovo_section(), "Add")
        # Hide the not-yet-implemented tabs (SAM/Membrane/Scribble/Histogram are
        # UI-only "Backend not connected" placeholders).  We still BUILD each
        # section above so its widgets stay parented (no top-level orphans) and
        # any attribute references keep working; we just don't expose the tab.
        # Indices are preserved by setTabVisible, so index-based lookups are safe.
        _hidden_tabs = {"SAM", "Membrane", "Scribble", "Histogram"}
        _default_idx = 0
        for _i in range(self._tool_tabs.count()):
            _name = self._tool_tabs.tabText(_i)
            if _name in _hidden_tabs:
                try:
                    self._tool_tabs.setTabVisible(_i, False)
                except Exception:
                    # Older Qt without setTabVisible — grey it out instead.
                    self._tool_tabs.setTabEnabled(_i, False)
            elif _name == "Reshape":
                _default_idx = _i
        self._tool_tabs.setCurrentIndex(_default_idx)
        root.addWidget(self._tool_tabs)

    def _apply_themed_styles(self) -> None:
        """Walk all child widgets and apply themed stylesheets to
        inputs + inline description labels.

        Applied ONLY to widgets without an existing stylesheet — so
        previously-styled widgets (Preview/Accept/Reject buttons, etc.)
        are preserved.  Centralizing here means theme.py updates
        propagate to every section without touching each
        ``_build_<section>`` method individually.
        """
        from qtpy.QtWidgets import (
            QComboBox as _QComboBox,
            QDoubleSpinBox as _QDoubleSpinBox,
            QLabel as _QLabel,
            QLineEdit as _QLineEdit,
            QSlider as _QSlider,
            QSpinBox as _QSpinBox,
        )
        # The Brick 2 mockup is the STEP 3 curation dock's design.  The
        # HITL 2D annotator is a different surface with its own review
        # history, so it keeps exactly the theme it had: same INPUT_STYLE,
        # no root stylesheet, no re-alignment.
        panel = self._mode != "hitl_2d"
        if panel:
            # Every tick on the panel is the mockup's `.cb` -- an 11 px
            # box that fills gold, with a tick in it, when it is on.  Set
            # on the ROOT so it reaches the ones built inside a stacked
            # page or a list as well.
            self.setStyleSheet(PANEL_CHECKBOX_STYLE)
        # Inputs — the panel's own field style on combos / spinboxes /
        # lineedits / sliders (the mockup's `.combo` and `.num`: a solid
        # surface, a small arrow, slim spin buttons).
        field_style = PANEL_INPUT_STYLE if panel else INPUT_STYLE
        for cls in (
            _QComboBox, _QSpinBox, _QDoubleSpinBox, _QLineEdit, _QSlider,
        ):
            for w in self.findChildren(cls):
                if not w.styleSheet():
                    w.setStyleSheet(field_style)
        # `.num` is right-aligned: a column of numbers is read down its
        # last digit, not its first.
        if panel:
            for cls in (_QSpinBox, _QDoubleSpinBox):
                for w in self.findChildren(cls):
                    try:
                        w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    except Exception:
                        pass
        # Description labels — many sections build inline-styled
        # "color: #aaa; font-size: 11px;" description QLabels.  Swap
        # them to the centralized DIM_LABEL_STYLE so the look matches
        # the annotator dock + Label Navigator.
        _INLINE_DIM = "color: #aaa; font-size: 11px;"
        _INLINE_NOTICE = "color: #777; font-style: italic;"
        for lbl in self.findChildren(_QLabel):
            ss = lbl.styleSheet().strip()
            if ss == _INLINE_DIM:
                lbl.setStyleSheet(DIM_LABEL_STYLE)
            elif ss == _INLINE_NOTICE:
                # Backend-not-connected placeholder text — keep distinct
                # from the regular dim label.
                lbl.setStyleSheet(DIM_LABEL_STYLE + " font-style: italic;")

    # --- SAM Assist (2D) ---------------------------------------------------

    def _build_sam_section(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        desc = QLabel(
            "SAM-assisted 2D annotation: paint FG/BG prompt points, "
            "get real-time mask predictions on the current slice."
        )
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.sam_enable_cb = QCheckBox("Enable SAM Assist")
        layout.addWidget(self.sam_enable_cb)

        # Device selection
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.sam_device_gpu = QRadioButton("GPU")
        self.sam_device_cpu = QRadioButton("CPU")
        self.sam_device_cpu.setChecked(True)
        device_row.addWidget(self.sam_device_gpu)
        device_row.addWidget(self.sam_device_cpu)
        device_row.addStretch(1)
        layout.addLayout(device_row)

        # Model selection (from registry)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.select_sam_model_combo = QComboBox()
        self._populate_sam_combo()
        model_row.addWidget(self.select_sam_model_combo, 1)
        layout.addLayout(model_row)

        self.load_sam_model_button = QPushButton("Load Model")
        layout.addWidget(self.load_sam_model_button)

        # Image layer for SAM encoder
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Image:"))
        self.sam_image_combo = QComboBox()
        img_row.addWidget(self.sam_image_combo, 1)
        layout.addLayout(img_row)

        # Accept/Reject
        btn_row = QHBoxLayout()
        self.sam_accept_btn = QPushButton("Accept")
        self.sam_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.sam_reject_btn = QPushButton("Reject")
        self.sam_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)
        btn_row.addWidget(self.sam_accept_btn)
        btn_row.addWidget(self.sam_reject_btn)
        layout.addLayout(btn_row)

        self.sam_status_label = QLabel("")
        self.sam_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.sam_status_label.setWordWrap(True)
        layout.addWidget(self.sam_status_label)

        notice = QLabel("Backend not connected — UI placeholder.")
        notice.setStyleSheet("color: #777; font-style: italic;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        return group

    def _populate_sam_combo(self) -> None:
        self.select_sam_model_combo.clear()
        for key, cfg in SAM_VARIANTS.items():
            self.select_sam_model_combo.addItem(cfg.get("label", key), key)

    # --- Guided Segmentation (Membrane Walker) -----------------------------

    def _build_membrane_walker_section(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        desc = QLabel(
            "Place membrane and interior seed points to guide a "
            "graph-based segmentation. Supports auto-snap, "
            "sensitivity presets, and handoff to other workflows."
        )
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Layer selectors
        form = QFormLayout()
        form.setVerticalSpacing(2)
        self.mw_image_combo = QComboBox()
        self.mw_target_combo = QComboBox()
        form.addRow("Image:", self.mw_image_combo)
        form.addRow("Target:", self.mw_target_combo)
        layout.addLayout(form)

        # Action buttons
        self.mw_start_btn = QPushButton("Start")
        self.mw_run_btn = QPushButton("Run Segmentation")
        self.mw_accept_btn = QPushButton("Accept")
        self.mw_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.mw_reject_btn = QPushButton("Reject")
        self.mw_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)

        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(self.mw_start_btn)
        btn_row1.addWidget(self.mw_run_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.mw_accept_btn)
        btn_row2.addWidget(self.mw_reject_btn)
        layout.addLayout(btn_row2)

        # Sensitivity preset
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity:"))
        self.mw_sensitivity_combo = QComboBox()
        self.mw_sensitivity_combo.addItems(["Conservative", "Balanced", "Aggressive"])
        self.mw_sensitivity_combo.setCurrentIndex(1)
        sens_row.addWidget(self.mw_sensitivity_combo, 1)
        layout.addLayout(sens_row)

        notice = QLabel("Backend not connected — UI placeholder.")
        notice.setStyleSheet("color: #777; font-style: italic;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        return group

    # --- Scribble Segmentation --------------------------------------------

    def _build_scribble_section(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        desc = QLabel(
            "Draw foreground/background scribbles, auto-solve via "
            "random walker. Supports undo, preview, and handoff."
        )
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Layer selectors
        form = QFormLayout()
        form.setVerticalSpacing(2)
        self.scribble_image_combo = QComboBox()
        self.scribble_target_combo = QComboBox()
        form.addRow("Image:", self.scribble_image_combo)
        form.addRow("Target:", self.scribble_target_combo)
        layout.addLayout(form)

        # Draw mode
        mode_row = QHBoxLayout()
        self.scribble_fg_btn = QPushButton("FG")
        self.scribble_fg_btn.setCheckable(True)
        self.scribble_fg_btn.setChecked(True)
        self.scribble_bg_btn = QPushButton("BG")
        self.scribble_bg_btn.setCheckable(True)
        mode_row.addWidget(self.scribble_fg_btn)
        mode_row.addWidget(self.scribble_bg_btn)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # Action buttons
        self.scribble_start_btn = QPushButton("Start")
        self.scribble_preview_btn = QPushButton("Preview")
        self.scribble_preview_btn.setStyleSheet(_REFINEMENT_PREVIEW_STYLE)
        self.scribble_accept_btn = QPushButton("Accept")
        self.scribble_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.scribble_reject_btn = QPushButton("Reject")
        self.scribble_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)

        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(self.scribble_start_btn)
        btn_row1.addWidget(self.scribble_preview_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.scribble_accept_btn)
        btn_row2.addWidget(self.scribble_reject_btn)
        layout.addLayout(btn_row2)

        notice = QLabel("Backend not connected — UI placeholder.")
        notice.setStyleSheet("color: #777; font-style: italic;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        return group

    # --- Histogram Segmentation -------------------------------------------

    def _build_histogram_section(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        desc = QLabel(
            "Interactive histogram with draggable FG/BG ranges. "
            "Supports auto-threshold (Otsu, multi-Otsu), random "
            "walker refinement, and handoff."
        )
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Layer selectors
        form = QFormLayout()
        form.setVerticalSpacing(2)
        self.hist_image_combo = QComboBox()
        self.hist_target_combo = QComboBox()
        form.addRow("Image:", self.hist_image_combo)
        form.addRow("Target:", self.hist_target_combo)
        layout.addLayout(form)

        # Threshold method
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Auto-threshold:"))
        self.hist_threshold_combo = QComboBox()
        self.hist_threshold_combo.addItems(["Otsu", "Multi-Otsu", "Manual"])
        thresh_row.addWidget(self.hist_threshold_combo, 1)
        layout.addLayout(thresh_row)

        # Transform
        transform_row = QHBoxLayout()
        transform_row.addWidget(QLabel("Transform:"))
        self.hist_transform_combo = QComboBox()
        self.hist_transform_combo.addItems([
            "None", "log1p", "Percentile", "Gamma", "HE", "AHE", "CLAHE",
        ])
        transform_row.addWidget(self.hist_transform_combo, 1)
        layout.addLayout(transform_row)

        # Action buttons
        self.hist_capture_roi_btn = QPushButton("Capture ROI")
        self.hist_preview_btn = QPushButton("Preview")
        self.hist_preview_btn.setStyleSheet(_REFINEMENT_PREVIEW_STYLE)
        self.hist_accept_btn = QPushButton("Accept")
        self.hist_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.hist_reject_btn = QPushButton("Reject")
        self.hist_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)

        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(self.hist_capture_roi_btn)
        btn_row1.addWidget(self.hist_preview_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.hist_accept_btn)
        btn_row2.addWidget(self.hist_reject_btn)
        layout.addLayout(btn_row2)

        notice = QLabel("Backend not connected — UI placeholder.")
        notice.setStyleSheet("color: #777; font-style: italic;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        return group

    # --- Refinement -------------------------------------------------------

    def _build_assisted_split_section(self) -> QWidget:
        """Build the Reshape page on the shared S1-S6 skeleton.

        Top to bottom, inside one scroll area: S1 STATUS, S2 TARGET,
        S3 SOURCES, S4 PREVIEW, S5 RESULT, then the ADVANCED disclosure --
        and, outside the scroll area but still a child of this page, the
        S6 FOOTER with exactly two buttons.

        Every widget is built HERE.  A stage change only flips container
        visibility, which is what the Add tab has always done and
        the Split page never did: its REVIEW rows used to be APPENDED below
        the Accept/Reject buttons that act on them (finding 2).
        """
        page, layout, footer, scroll = self._page_with_footer()
        content = scroll.widget()

        # -- header: the tool's name + its `?` ----------------------------
        # The guide covers BOTH the long-standing manual split and the
        # guided modes, because from the user's side they are one tool with
        # two pickers.
        _hdr = QHBoxLayout()
        _hdr_label = QLabel("Reshape an object")
        _hdr_label.setStyleSheet(SECTION_HEADER_STYLE)
        _hdr.addWidget(_hdr_label)
        _hdr.addStretch(1)
        _hdr.addWidget(_help_button(
            content,
            guide_path=str(_GUIDES_DIR / "annotator_3d_reshape.md"),
            title="Reshape an object -- 3D Annotator",
            tooltip="Help: Reshape",
        ))
        layout.addLayout(_hdr)

        # ONE line, and what is behind it.  The 10-line
        # workflow + method essay that used to head this page wrapped to
        # roughly 20 visual lines at the dock's 340 px and was the first
        # thing the user scrolled past every time (finding 5).  This is
        # the opposite trade: one sentence that is always readable, and
        # four paragraphs that are there when someone wants them.
        #
        # It REPLACES `_split_desc_label` in place -- same slot in the
        # header block, above S1 -- so the S1..S6 order does not move and
        # nothing is appended after construction.
        # A QPushButton cannot word-wrap, and its sizeHint is the width
        # of the whole sentence -- one line of 110 characters asks for
        # 759 px in a 316 px viewport, and the horizontal scrollbar is
        # off, so it would be CLIPPED rather than scrolled to.  The
        # sentence is therefore a wrapping label BESIDE the chevron, as
        # designed, and the label forwards its clicks so the
        # whole line is one disclosure.
        _about_row = QHBoxLayout()
        _about_row.setContentsMargins(0, 0, 0, 0)
        _about_row.setSpacing(4)
        self.split_about_btn = QPushButton(_CHEVRON_CLOSED, content)
        self.split_about_btn.setCheckable(True)
        self.split_about_btn.setChecked(False)
        self.split_about_btn.setStyleSheet(PANEL_CHEVRON_STYLE)
        self.split_about_btn.setToolTip(
            "What this tool does, in plain words.")
        self.split_about_btn.setFixedWidth(_about_btn_width(self.split_about_btn))
        _about_row.addWidget(self.split_about_btn, 0, Qt.AlignTop)
        self._split_about_line = _ClickableLabel(
            self.split_about_btn.toggle, content)
        self._split_about_line.setText(SPLIT_ABOUT_LINE)
        self._split_about_line.setStyleSheet(
            PANEL_HELP_STYLE + _ClickableLabel.HOVER_STYLE)
        self._split_about_line.setWordWrap(True)
        self._split_about_line.setToolTip(
            "What this tool does, in plain words.")
        _about_row.addWidget(self._split_about_line, 1)
        layout.addLayout(_about_row)

        self._split_about_body = QLabel(content)
        # RICH TEXT (M2): `<b>` for the two inputs, `&#8212;` for the
        # asides.  Qt would auto-detect the markup anyway, but saying so
        # makes the paragraph split below correct BY CONSTRUCTION rather
        # than by luck -- a blank line means nothing in rich text, so the
        # module constant's "\n\n" is turned into real breaks here and
        # the constant stays readable (and diffable) as plain paragraphs.
        self._split_about_body.setTextFormat(Qt.RichText)
        self._split_about_body.setText(
            SPLIT_ABOUT_BODY.replace("\n\n", "<br><br>"))
        self._split_about_body.setStyleSheet(PANEL_HELP_STYLE)
        self._split_about_body.setWordWrap(True)
        self._split_about_body.setVisible(False)
        layout.addWidget(self._split_about_body)
        self.split_about_btn.toggled.connect(self._on_split_about_toggled)

        # ==================================================================
        # S1 STATUS -- one line, always present, never hidden.
        # ==================================================================
        self.split_status_label = QLabel("Select an object, then press Start.")
        self.split_status_label.setStyleSheet(_PANEL_STATUS_TEXT_STYLE)
        self.split_status_label.setWordWrap(True)
        self._split_s1_container = self._status_row(
            "_split_status_dot", self.split_status_label
        )
        layout.addWidget(self._split_s1_container)

        # ==================================================================
        # S2 TARGET -- the cell, and the only door into a session.
        # ==================================================================
        self._split_s2_container = QWidget()
        _s2 = QVBoxLayout(self._split_s2_container)
        _s2.setContentsMargins(0, 0, 0, 0)
        _s2.setSpacing(2)

        # ONE object, two labels.  "Change" is not a second
        # button: the target is punched out of the volume from Start
        # onwards, so re-targeting would abandon an unrestored hole -- it is
        # the same button, disabled, saying what it would do if it could.
        self.split_start_btn = QPushButton("Start")
        self.split_start_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_start_btn.setToolTip(
            "Hold the selected object out of the volume and begin a reshape. "
            "The object stays hidden until you Accept or Discard."
        )
        _s2.addWidget(self.split_start_btn)

        # Brick U2.  One checkbox and one property set -- no
        # isolation, no sync, no rendering code is touched.  Setting
        # `contour` on the Refine layer reaches `Refine_XY/_XZ/_YZ`
        # through the propagation `sync_label_properties` has always
        # done, and the main `curation_3d_*` layer is provably out of
        # reach of it (different `_base_label_name`).
        self.split_contour_cb = QCheckBox(
            "Show this object as an outline", self._split_s2_container)
        self.split_contour_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.split_contour_cb.setToolTip(HELP_LINES["split_contour_cb"])
        _s2.addWidget(self.split_contour_cb)
        self._split_contour_help = self._helper_line("split_contour_cb")
        _s2.addWidget(self._split_contour_help)

        # Only true before Start: once the cell is held out of the
        # volume it is picked, and the line is 27 px of the 247 px
        # viewport telling the user to do what they just did.
        self._split_target_hint = QLabel(
            "Pick the object in the 3D view or the Label Navigator."
        )
        self._split_target_hint.setWordWrap(True)
        self._split_target_hint.setStyleSheet(PANEL_HELP_STYLE)
        _s2.addWidget(self._split_target_hint)
        layout.addWidget(self._split_s2_container)

        # ==================================================================
        # S3 SOURCES -- one collapsed summary line that opens IN PLACE, and
        # the seed work rows, which are not behind the disclosure.
        # ==================================================================
        self._split_s3_container = QWidget()
        _s3 = QVBoxLayout(self._split_s3_container)
        _s3.setContentsMargins(0, 0, 0, 0)
        _s3.setSpacing(2)

        # ------------------------------------------------------------------
        # Guided Split -- "Seeds from" / "Walls from".
        #
        # Two pickers, COLLAPSED behind a one-line summary, both opening on
        # today's behaviour.  Touch nothing and the split below behaves
        # exactly as it always has -- the orchestrator's guided branch is
        # gated on `is_guided_split()` returning False at the defaults.
        # ------------------------------------------------------------------
        self.split_guided_summary_btn = QPushButton()
        self.split_guided_summary_btn.setCheckable(True)
        self.split_guided_summary_btn.setChecked(False)
        self.split_guided_summary_btn.setStyleSheet(PANEL_CHEVRON_STYLE)
        self.split_guided_summary_btn.setProperty(
            _ELIDE_HINT_PROP,
            "Choose where the markers and the walls come from. "
            "The defaults are what this tool has always done.",
        )
        self.split_guided_summary_btn.setToolTip(
            "Choose where the markers and the walls come from. "
            "The defaults are what this tool has always done."
        )
        _s3.addWidget(self.split_guided_summary_btn)

        self._split_guided_container = QWidget()
        _g = QVBoxLayout(self._split_guided_container)
        # The cards below carry the inset now, so this column has none:
        # every px of padding a card adds has to be bought back from the
        # rhythm around it (the markers-found state has 1 px in hand).
        _g.setContentsMargins(0, 0, 0, 0)
        _g.setSpacing(2)

        # ---- the three sibling sub-sections ----------------
        # Markers / Walls / How to cut, only ONE open at a time -- the
        # Add tab's container-swap pattern, not a new one.  Three open
        # forms do not fit the 247 px viewport, and the one the user is
        # working in has to stay above Preview.
        self.split_markers_btn = self._subsection_chevron(
            "Markers",
            "Where the markers come from - one marker per object you expect to come out.",
        )
        _g.addWidget(self.split_markers_btn)
        self._split_markers_container = QWidget()
        _mk = QVBoxLayout(self._split_markers_container)
        _g.addWidget(self._split_markers_container)
        # The mockup's `.sec`: the open sub-section is a bordered card
        # whose header is the chevron above it.  Only the OPEN one is a
        # card -- three closed cards would cost 36 px of a viewport that
        # has 1 px spare in the markers-found state.
        make_panel_card(self._split_markers_container)

        # -- Markers from ----------------------------------------------
        _seeds_row = QHBoxLayout()
        _seeds_row.setSpacing(6)
        _seeds_row.addWidget(panel_form_label("Markers from", parent=self))
        self.split_seeds_from_combo = QComboBox()
        self.split_seeds_from_combo.addItems(list(SEEDS_FROM_LABELS))
        self.split_seeds_from_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seeds_from_combo.setToolTip(
            "Where the markers come from -- one marker per object you expect "
            "to come out.  Each choice explains itself in the grey line "
            "under this row."
        )
        _seeds_row.addWidget(self.split_seeds_from_combo, 1)
        _mk.addLayout(_seeds_row)
        _mk.addWidget(self._item_help_label("split_seeds_from_combo"))

        # -- the guide picker (only for the two guided sources) ------------
        self._split_guide_container = QWidget()
        _gc = QVBoxLayout(self._split_guide_container)
        _gc.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _gc.setSpacing(3)

        # The RUN choice and the FILE choice are two different questions
        # (finding 9): index 2 wants a run, index 3 wants a file on disk.
        # Each lives in its own row so only the one that applies is shown.
        self._split_guide_run_row = QWidget()
        _grr = QHBoxLayout(self._split_guide_run_row)
        _grr.setContentsMargins(0, 0, 0, 0)
        _grr.setSpacing(0)
        self.split_guide_source_combo = QComboBox()
        self.split_guide_source_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_guide_source_combo.setToolTip(
            "Which set of labels to trust as the markers. Each row names the "
            "run, what it holds, how far that run got on this patch, and "
            "which file is being read."
        )
        _grr.addWidget(self.split_guide_source_combo, 1)
        _gc.addWidget(self._split_guide_run_row)

        _guide_btn_row = QHBoxLayout()
        self._split_guide_file_row = QWidget()
        _gfr = QHBoxLayout(self._split_guide_file_row)
        _gfr.setContentsMargins(0, 0, 0, 0)
        _gfr.setSpacing(0)
        self.split_guide_browse_btn = QPushButton("Browse...")
        self.split_guide_browse_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_guide_browse_btn.setToolTip(
            "Pick a mask TIFF from anywhere on disk."
        )
        _gfr.addWidget(self.split_guide_browse_btn)
        _guide_btn_row.addWidget(self._split_guide_file_row)
        self.split_guide_load_btn = QPushButton("Load mask")
        self.split_guide_load_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_guide_load_btn.setToolTip(
            "Read the chosen mask and show it over the volume. It is never "
            "edited -- it is only used to place the markers."
        )
        _guide_btn_row.addWidget(self.split_guide_load_btn)
        _gc.addLayout(_guide_btn_row)

        self.split_guide_show_cb = QCheckBox("Show guide mask")
        self.split_guide_show_cb.setChecked(True)
        self.split_guide_show_cb.setEnabled(False)
        self.split_guide_show_cb.setToolTip(
            "Show or hide the trusted mask. It has no keyboard shortcut "
            "because it is not one of the layers you are curating."
        )
        _gc.addWidget(self.split_guide_show_cb)

        self.split_guide_note_label = QLabel("")
        self.split_guide_note_label.setWordWrap(True)
        self.split_guide_note_label.setStyleSheet(PANEL_WARN_STYLE)
        self.split_guide_note_label.setVisible(False)
        _gc.addWidget(self.split_guide_note_label)

        self._split_guide_container.setVisible(False)
        _mk.addWidget(self._split_guide_container)

        # -- the image-channel marker block (seed source index 4) ------
        self._split_image_seed_container = QWidget()
        _isc = QVBoxLayout(self._split_image_seed_container)
        _isc.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _isc.setSpacing(2)

        _sch_row = QHBoxLayout()
        _sch_row.setSpacing(6)
        _sch_row.addWidget(panel_form_label("Channel", parent=self))
        self.split_seed_channel_combo = QComboBox()
        self.split_seed_channel_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_channel_combo.setToolTip(
            "The image channel the markers are found in -- usually the "
            "nuclei stain.  Blank means no channel is chosen: the tool "
            "says so rather than guessing one."
        )
        _sch_row.addWidget(self.split_seed_channel_combo, 1)
        _isc.addLayout(_sch_row)

        _det_row = QHBoxLayout()
        _det_row.setSpacing(6)
        _det_row.addWidget(panel_form_label("Find by", parent=self))
        self.split_seed_detector_combo = QComboBox()
        self.split_seed_detector_combo.addItems(list(SEED_DETECTOR_LABELS))
        self.split_seed_detector_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_detector_combo.setToolTip(
            "How the markers are found in that channel.  Nucleus peaks is the\n"
            "usual answer; Membrane-enclosed interiors reads the MEMBRANE\n"
            "channel instead and finds each dim space the membrane encloses."
        )
        _det_row.addWidget(self.split_seed_detector_combo, 1)
        _isc.addLayout(_det_row)
        _isc.addWidget(self._item_help_label("split_seed_detector_combo"))

        # -- Prepare the channel first (one tick per row) -----
        self.split_seed_prep_btn = self._subsection_chevron(
            "Prepare the channel first: local contrast",
            "Clean the channel up before the markers are looked for.",
        )
        self.split_seed_prep_btn.setProperty("_epicycle_elide_title", True)
        # The mockup's `.chip`: a small flat toggle that tints gold while
        # it is open, not a third level of section.
        self.split_seed_prep_btn.setStyleSheet(PANEL_CHIP_STYLE)
        _isc.addWidget(self.split_seed_prep_btn)
        self._split_seed_prep_container = QWidget()
        _spc = QVBoxLayout(self._split_seed_prep_container)
        _spc.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _spc.setSpacing(1)
        for _key, _attr, _on in (
            ("clahe", "split_seed_prep_clahe_cb", True),
            ("zcorrect", "split_seed_prep_zcorrect_cb", False),
            ("bgsub", "split_seed_prep_bgsub_cb", False),
            ("lift", "split_seed_prep_lift_cb", False),
            ("denoise", "split_seed_prep_denoise_cb", False),
        ):
            _cb = QCheckBox(PREP_TICK_LABELS[_key])
            _cb.setChecked(_on)
            _cb.setToolTip(HELP_LINES[_attr])
            setattr(self, _attr, _cb)
            _spc.addWidget(_cb)
            _spc.addWidget(self._helper_line(_attr))
        _order = QLabel(_PREP_ORDER_SENTENCE)
        _order.setWordWrap(True)
        _order.setStyleSheet(_HELPER_LINE_STYLE)
        _spc.addWidget(_order)
        self._split_seed_prep_container.setVisible(False)
        _isc.addWidget(self._split_seed_prep_container)

        self._split_image_seed_container.setVisible(False)
        _mk.addWidget(self._split_image_seed_container)

        # -- Find markers + the cherry-pick ticklist ---------
        self._split_find_container = QWidget()
        _fc = QVBoxLayout(self._split_find_container)
        _fc.setContentsMargins(0, 0, 0, 0)
        _fc.setSpacing(2)

        _find_row = QHBoxLayout()
        _find_row.setSpacing(5)
        self.split_find_markers_btn = QPushButton("Find markers")
        self.split_find_markers_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_find_markers_btn.setToolTip(
            "Look for the markers now and list them, so you can untick "
            "anything that is not an object.  Nothing is cut."
        )
        _find_row.addWidget(self.split_find_markers_btn)
        self.split_marker_count_label = QLabel("")
        self.split_marker_count_label.setWordWrap(True)
        self.split_marker_count_label.setStyleSheet(PANEL_HELP_STYLE)
        _find_row.addWidget(self.split_marker_count_label, 1)
        _fc.addLayout(_find_row)
        # The LIST this row fills does not live here any more: it is ONE list, shared with the painted sources, and it
        # sits in the session block below so hand-painting can see it
        # without opening this disclosure.

        self._split_find_container.setVisible(False)
        _mk.addWidget(self._split_find_container)

        # ---- Walls ----------------------------------------------------
        self.split_walls_btn = self._subsection_chevron(
            "Walls",
            "What the cut is allowed to follow.",
        )
        _g.addWidget(self.split_walls_btn)
        self._split_walls_group_container = QWidget()
        _wl = QVBoxLayout(self._split_walls_group_container)
        _g.addWidget(self._split_walls_group_container)
        make_panel_card(self._split_walls_group_container)

        _walls_row = QHBoxLayout()
        _walls_row.setSpacing(6)
        _walls_row.addWidget(panel_form_label("Walls from", parent=self))
        self.split_walls_from_combo = QComboBox()
        self.split_walls_from_combo.addItems(list(WALLS_FROM_LABELS))
        self.split_walls_from_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_walls_from_combo.setToolTip(
            "What the cut is allowed to follow.  Each choice explains "
            "itself in the grey line under this row; the ones that read "
            "an image channel show a channel picker below."
        )
        _walls_row.addWidget(self.split_walls_from_combo, 1)
        _wl.addLayout(_walls_row)
        _wl.addWidget(self._item_help_label("split_walls_from_combo"))

        self._split_walls_container = QWidget()
        _wc = QVBoxLayout(self._split_walls_container)
        _wc.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _wc.setSpacing(3)

        _wch_row = QHBoxLayout()
        _wch_row.setSpacing(6)
        _wch_row.addWidget(panel_form_label("channel", parent=self))
        self.split_wall_channel_combo = QComboBox()
        self.split_wall_channel_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_wall_channel_combo.setToolTip(
            "The image channel the walls are read from.  Blank means no "
            "channel is chosen: the tool says so rather than guessing one."
        )
        _wch_row.addWidget(self.split_wall_channel_combo, 1)
        _wc.addLayout(_wch_row)

        self._split_walls_container.setVisible(False)
        _wl.addWidget(self._split_walls_container)

        # -- Prepare the WALL channel ('Same as markers' is a
        # MODE, not a step -- while it is ticked the four below mirror
        # the marker chain and are greyed).
        self.split_wall_prep_btn = self._subsection_chevron(
            "Prepare the channel first: none",
            "Clean the wall channel up before the cut follows it.",
        )
        self.split_wall_prep_btn.setProperty("_epicycle_elide_title", True)
        self.split_wall_prep_btn.setStyleSheet(PANEL_CHIP_STYLE)
        _wl.addWidget(self.split_wall_prep_btn)
        self._split_wall_prep_container = QWidget()
        _wpc = QVBoxLayout(self._split_wall_prep_container)
        _wpc.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _wpc.setSpacing(1)
        self.split_wall_prep_same_cb = QCheckBox("Same as markers")
        self.split_wall_prep_same_cb.setToolTip(
            HELP_LINES["split_wall_prep_same_cb"]
        )
        _wpc.addWidget(self.split_wall_prep_same_cb)
        _wpc.addWidget(self._helper_line("split_wall_prep_same_cb"))
        for _key, _attr in (
            ("zcorrect", "split_wall_prep_zcorrect_cb"),
            ("bgsub", "split_wall_prep_bgsub_cb"),
            ("lift", "split_wall_prep_lift_cb"),
            ("denoise", "split_wall_prep_denoise_cb"),
        ):
            _cb = QCheckBox(PREP_TICK_LABELS[_key])
            _cb.setToolTip(HELP_LINES[_attr])
            setattr(self, _attr, _cb)
            _wpc.addWidget(_cb)
            _wpc.addWidget(self._helper_line(_attr))
        _worder = QLabel(_PREP_ORDER_SENTENCE)
        _worder.setWordWrap(True)
        _worder.setStyleSheet(_HELPER_LINE_STYLE)
        _wpc.addWidget(_worder)
        self._split_wall_prep_container.setVisible(False)
        _wl.addWidget(self._split_wall_prep_container)

        # ---- How to cut (Solve with was promoted OUT of Advanced) -----
        self.split_solve_btn = self._subsection_chevron(
            "How to cut",
            "How the cut itself is computed.",
        )
        _g.addWidget(self.split_solve_btn)
        self._split_solve_container = QWidget()
        _sv = QVBoxLayout(self._split_solve_container)
        _g.addWidget(self._split_solve_container)
        make_panel_card(self._split_solve_container)

        _solver_row = QHBoxLayout()
        _solver_row.setSpacing(6)
        _solver_row.addWidget(panel_form_label("Solve with", parent=self))
        self.split_solver_combo = QComboBox()
        self.split_solver_combo.addItems(list(SPLIT_SOLVER_LABELS))
        self.split_solver_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_solver_combo.setToolTip(
            "How the cut is placed.\n\n"
            "Sharp cut: floods out from each marker and cuts on the highest "
            "ridge between them.\n"
            "Soft cut: walks probabilities out from the markers, so the cut "
            "eases onto a faint wall instead of snapping to the nearest "
            "strong one.\n"
            "Settle onto the wall signal: starts from the sharp cut, then "
            "lets the labels spread through the image so the boundary slides "
            "onto the wall signal. IT IS NOT SMOOTHING -- the number of "
            "pieces is "
            "preserved and the result can look rougher, not smoother. It "
            "needs u-Segment3D and falls back to the sharp cut with a note "
            "if that is missing."
        )
        _solver_row.addWidget(self.split_solver_combo, 1)
        _sv.addLayout(_solver_row)
        _sv.addWidget(self._item_help_label("split_solver_combo"))

        # ---- the three promoted rows (parity rows 54/60) -----
        # Snap and the expected count were Advanced ticks; the arbiter
        # was a row inside the Walls card, two chevrons away from the
        # solver it modifies.  All three answer "how should the cut come
        # out", which is the question this card asks.
        #
        # The first two need an image-derived wall as evidence, so they
        # live in their own container and disappear together with it.
        self._split_cut_image_container = QWidget()
        _cic = QVBoxLayout(self._split_cut_image_container)
        _cic.setContentsMargins(0, 0, 0, 0)
        _cic.setSpacing(1)

        self.split_refine_guided_cb = QCheckBox("Snap to the wall signal")
        self.split_refine_guided_cb.setToolTip(
            "Needs u-Segment3D. After the cut, slides each piece's edge "
            "onto the nearest edge in the wall channel. Skipped with a "
            "note if it would erase a piece."
        )
        _cic.addWidget(self.split_refine_guided_cb)
        _cic.addWidget(self._helper_line("split_refine_guided_cb"))

        self.split_arbiter_cb = QCheckBox("Only cut where I can see a wall")
        self.split_arbiter_cb.setChecked(False)
        self.split_arbiter_cb.setToolTip(
            "With this on, a proposed cut is kept only where there really "
            "is a wall between the two markers. Where there is none, the "
            "object is left whole and marked as possibly dividing -- and you "
            "can still force the cut with Split anyway.\n\n"
            "It is off by default: a cut you asked for is not "
            "second-guessed unless you opt in.\n\n"
            "The test is relative to this object's own brightness, not an "
            "absolute level."
        )
        _cic.addWidget(self.split_arbiter_cb)
        _cic.addWidget(self._helper_line("split_arbiter_cb"))
        self._split_cut_image_container.setVisible(False)
        _sv.addWidget(self._split_cut_image_container)

        # The expected count is about the ANSWER, not about the image, so
        # it is shown whatever the walls are.
        _count_row = QHBoxLayout()
        _count_row.setSpacing(6)
        #  draws this as "I expect exactly [ 2 ] cells", and the
        # words have to be SPLIT around the number to be that sentence.
        # They also have to be split to fit: as one 262 px label with the
        # spin box beside it the row asked for 358 px in a 316 px
        # viewport, and the horizontal scrollbar is off -- so opening
        # How-to-cut silently clipped the whole page's right edge, Start
        # and Preview and this spin box with it.
        self.split_target_count_cb = QCheckBox("I expect exactly")
        self.split_target_count_cb.setToolTip(
            "If more pieces come out than this, the weakest joins are "
            "merged until the count matches. If FEWER come out you get a "
            "warning and nothing is forced."
        )
        _count_row.addWidget(self.split_target_count_cb)
        self.split_target_count_sb = QSpinBox()
        # 2 is the smallest count worth asking for: 0 and 1 used to mean
        # "off", and the tick means that now.
        self.split_target_count_sb.setRange(2, 100)
        self.split_target_count_sb.setValue(2)
        self.split_target_count_sb.setEnabled(False)
        self.split_target_count_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_target_count_sb.setToolTip(
            "How many objects you expect this one to become."
        )
        _count_row.addWidget(self.split_target_count_sb, 1)
        self.split_target_count_unit = panel_form_label(
            "objects", 42, self)
        _count_row.addWidget(self.split_target_count_unit)
        _sv.addLayout(_count_row)
        _sv.addWidget(self._helper_line("split_target_count_cb"))
        self.split_target_count_cb.toggled.connect(
            self.split_target_count_sb.setEnabled
        )

        # The 2024 manual pair (`split_method_combo` /
        # `split_image_combo`) asked the SAME question as "Walls
        # from" one chevron above, and has been hidden since Brick 1.
        # Retired in Brick 2b: its manual branch was
        # UI-unreachable, because `is_guided_split()` routes every
        # wall index >= 1 into the engine -- pinned by a truth table
        # over both pickers, not by a call recorder on dead code.

        self._split_guided_container.setVisible(False)
        _s3.addWidget(self._split_guided_container)

        # -- the PAINTED-seed work rows ---------------------------------
        # They are about painting markers by hand, so they are hidden
        # wherever the tool finds its own (the same rule, and the same
        # set, that hides Detect Seeds).  Leaving them on screen cost
        # ~150 px of the 247 px viewport and told the user to paint
        # markers that were about to be found for them.
        self._split_paint_rows_container = QWidget()
        _pr_rows = QVBoxLayout(self._split_paint_rows_container)
        _pr_rows.setContentsMargins(0, 0, 0, 0)
        _pr_rows.setSpacing(3)
        _s3_outer, _s3 = _s3, _pr_rows
        # The seed count and the distance-heatmap toggle share one row:
        # both are one short thing, and on two rows they cost 21 px of
        # the height that decides whether Preview is above the fold at
        # the live 209 px viewport (owner, 2026-09-18).
        _seed_head = QHBoxLayout()
        _seed_head.setSpacing(6)
        # "Markers", never "seeds", in anything the user reads.  This
        # is the line `seed_count=` rewrites; its BIRTH text is what a
        # first-time user reads before anything has been pushed.
        self.split_seed_label = QLabel("Markers: 0")
        self.split_seed_label.setStyleSheet(_PANEL_STATUS_TEXT_STYLE)
        _seed_head.addWidget(self.split_seed_label)
        _seed_head.addStretch(1)

        # EDT overlay checkbox -- disabled until Start Split is clicked.
        self.split_edt_cb = QCheckBox("Show distance heatmap")
        self.split_edt_cb.setEnabled(False)
        self.split_edt_cb.setToolTip(
            "Colour the object by how deep inside it each voxel is. "
            "Bright means far from every edge; put one marker in each "
            "separate bright lobe."
        )
        _seed_head.addWidget(self.split_edt_cb)
        _s3.addLayout(_seed_head)

        seed_btn_row = QHBoxLayout()
        seed_btn_row.setSpacing(4)
        self.split_next_seed_btn = QPushButton("Add marker")
        # Chips, not buttons: three full-weight buttons on this one row
        # need 313 px of the viewport's 316, and the page is clipped
        # rather than scrolled past it.
        self.split_next_seed_btn.setStyleSheet(PANEL_CHIP_STYLE)
        self.split_next_seed_btn.setEnabled(False)
        self.split_next_seed_btn.setToolTip(
            "Start a new marker for the next object"
        )
        seed_btn_row.addWidget(self.split_next_seed_btn)

        self.split_undo_seed_btn = QPushButton("Undo")
        self.split_undo_seed_btn.setStyleSheet(PANEL_CHIP_STYLE)
        self.split_undo_seed_btn.setEnabled(False)
        self.split_undo_seed_btn.setToolTip(
            "Remove the last marker and go back to painting the previous one"
        )
        seed_btn_row.addWidget(self.split_undo_seed_btn)

        # Detect Seeds button -- now in the seed row, where the other two
        # ways of making a seed already are.
        # Source 0's finder, named after what it LOOKS AT rather than
        # what the code calls its output (fix pass M3).
        self.split_detect_seeds_btn = QPushButton("Find from shape")
        self.split_detect_seeds_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_detect_seeds_btn.setEnabled(False)
        self.split_detect_seeds_btn.setToolTip(
            "Place the markers for you, at the thickest points inside "
            "the object. Drag the line on the little chart to keep more "
            "or fewer."
        )
        seed_btn_row.addWidget(self.split_detect_seeds_btn)
        # ... and nothing stretches: a chip that fills a third of the dock
        # is not a chip.
        seed_btn_row.addStretch(1)
        _s3.addLayout(seed_btn_row)

        # --- Auto-detect histogram container (initially hidden) ---
        self._detect_container = QWidget()
        detect_layout = QVBoxLayout(self._detect_container)
        detect_layout.setContentsMargins(0, 2, 0, 2)
        detect_layout.setSpacing(2)

        # Min distance spinbox
        # One knob per row, on the same label column as every other row
        # on this page.  Both on ONE row starved them: with the shrink
        # policy that keeps Advanced from clipping the page, the HBox
        # took "Min dist" down to its minus button and no number at all
        # (owner screenshot, 2026-09-18).  Per-row also leaves the space
        # Brick 2b's helper lines need under each.
        detect_params_row = QHBoxLayout()
        detect_params_row.setSpacing(6)
        detect_params_row.addWidget(
            panel_form_label("Min dist:", parent=self))
        self.split_detect_min_dist_sb = QSpinBox()
        self.split_detect_min_dist_sb.setRange(2, 20)
        self.split_detect_min_dist_sb.setValue(5)
        self.split_detect_min_dist_sb.setToolTip(
            "How far apart, in voxels, two thick points have to be to "
            "count as two markers. Lower = more markers, higher = fewer. "
            "Start with 5 and go lower if the cut misses an object."
        )
        detect_params_row.addWidget(self.split_detect_min_dist_sb, 1)
        detect_layout.addLayout(detect_params_row)
        detect_layout.addWidget(self._helper_line("split_detect_min_dist_sb"))

        detect_bins_row = QHBoxLayout()
        detect_bins_row.setSpacing(6)
        detect_bins_row.addWidget(panel_form_label("Bins:", parent=self))
        self.split_detect_bins_sb = QSpinBox()
        self.split_detect_bins_sb.setRange(3, 50)
        self.split_detect_bins_sb.setValue(15)
        self.split_detect_bins_sb.setToolTip(
            "Number of histogram bins for Euclidean Distance Transform (EDT) "
            "peak distribution. More bins = finer control over threshold."
        )
        detect_bins_row.addWidget(self.split_detect_bins_sb, 1)
        detect_layout.addLayout(detect_bins_row)
        detect_layout.addWidget(self._helper_line("split_detect_bins_sb"))

        # Wire bins spinbox to live-redraw histogram.
        self._detect_last_edt_values: list[float] = []
        self.split_detect_bins_sb.valueChanged.connect(self._on_bins_changed)

        # pyqtgraph histogram (delayed import)
        self._detect_plot_widget = None
        self._detect_bar_item = None
        self._detect_thresh_line = None
        self._detect_plot_placeholder = QWidget()
        self._detect_plot_placeholder.setFixedHeight(100)
        detect_layout.addWidget(self._detect_plot_placeholder)
        # The draggable line IS the control here; the owner found varying
        # it the productive part of the row, so it gets its own sentence.
        detect_layout.addWidget(self._helper_line("_detect_plot_placeholder"))

        # Detected count label
        self._detect_count_label = QLabel("Found: 0 markers")
        self._detect_count_label.setStyleSheet(PANEL_HELP_STYLE)
        detect_layout.addWidget(self._detect_count_label)

        self._detect_container.setVisible(False)
        _s3.addWidget(self._detect_container)

        # ------------------------------------------------------------------
        # THE marker list.
        #
        # ONE list for all five sources.  It used to be two -- a painted
        # `split_seed_list` here and a tickable `split_marker_list` up in
        # the Markers card -- so three of the five sources had no way to
        # untick anything, and the two lists were read as the same thing.
        # `split_seed_list` is RETIRED; `seed_ids=` / `active_seed_id=`
        # now point here, so no orchestrator call site moved.
        #
        # It lives in the SESSION block rather than inside the Sources
        # disclosure, because hand-painting has to show its markers
        # without the user opening a chevron first (the painting rows are
        # here for the same reason).
        # ------------------------------------------------------------------
        # The caption and the two chips share ONE row.  Both lists are on
        # screen at once now, so the caption is not optional -- but the
        # markers-found state gained the whole painting block this round
        # and every row has to earn its height.
        _marker_head = QHBoxLayout()
        _marker_head.setSpacing(4)
        self.split_marker_caption = QLabel(
            "Markers - before the cut. Decides what the cut is made from."
        )
        # ELIDED, not wrapped: at the 316 px viewport it is two lines,
        # and the second one costs a row of markers.  The whole sentence
        # is in the tooltip, as the Brick 2a caption's was.
        self.split_marker_caption.setWordWrap(False)
        self.split_marker_caption.setToolTip(
            "Markers - before the cut. Decides what the cut is made from."
        )
        self.split_marker_caption.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.split_marker_caption.setStyleSheet(PANEL_HELP_STYLE)
        self.split_marker_caption.setVisible(False)
        _marker_head.addWidget(self.split_marker_caption, 1)

        # All / None ride the list they act on, not the row that produced
        # it: source 0 has no Find row at all.
        self.split_marker_all_btn = QPushButton("All")
        self.split_marker_all_btn.setStyleSheet(PANEL_CHIP_STYLE)
        self.split_marker_all_btn.setToolTip(
            "Tick every marker, which puts every lifted marker back on "
            "the object."
        )
        self.split_marker_all_btn.setVisible(False)
        self.split_marker_none_btn = QPushButton("None")
        self.split_marker_none_btn.setStyleSheet(PANEL_CHIP_STYLE)
        self.split_marker_none_btn.setToolTip(
            "Untick every marker, which lifts them all off the object.  "
            "Preview then refuses rather than cutting blind."
        )
        self.split_marker_none_btn.setVisible(False)
        _marker_head.addWidget(self.split_marker_all_btn)
        _marker_head.addWidget(self.split_marker_none_btn)
        _s3.addLayout(_marker_head)

        self.split_marker_list = QListWidget()
        # The cap.  It is a CAP, not a reserve: the list is as
        # tall as the markers in it (`_fit_panel_list`) and only ever
        # scrolls past this -- on the live dock a row is 26 px, so 76 px
        # was cutting the third row in half.
        self.split_marker_list.setMaximumHeight(_MARKER_LIST_MAX_H)
        self._style_panel_list(self.split_marker_list)
        # The ONE visual difference from the piece list: a gold left edge
        # on every marker row.  A caption alone was not enough -- the two
        # lists look identical and are three slots apart (the two
        # lists must never be confusable).
        self.split_marker_list.setStyleSheet(_MARKER_LIST_STYLE)
        self.split_marker_list.setToolTip(
            "Every marker in this object.  Preview cuts with the ticked "
            "ones; unticking a marker means \"that is not an object\" "
            "and lifts it off the object -- it disappears here and in "
            "the side views.  Tick it again to put it back (if you "
            "painted another marker over it meanwhile, that marker keeps "
            "the overlap)."
        )
        self.split_marker_list.setVisible(False)
        _s3.addWidget(self._list_with_grip(
            self.split_marker_list, "split_marker_list",
            _MARKER_LIST_H_PROP, _MARKER_LIST_MAX_H))

        # The grey line that used to sit here -- "Unticked markers are
        # lifted off the object - tick again to put them back." -- is
        # GONE (audit H2, Brick U4).  With the hand-off button on screen
        # in guided review the page measured 778 px against its 753 px
        # ceiling, and this line is the one control whose job the page
        # now does by itself: since U2b the marker visibly leaves the
        # object and all three side views the moment you untick it.  The
        # sentence lives on in the list's own tooltip and on the None
        # chip's, where it is read by the people who have not worked it
        # out yet, at no cost in height.

        self.split_refine_seeds_btn = QPushButton("Edit markers")
        self.split_refine_seeds_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_refine_seeds_btn.setEnabled(False)
        self.split_refine_seeds_btn.setToolTip(
            "Go back to editing the markers before previewing again"
        )
        _s3.addWidget(self.split_refine_seeds_btn)

        self._split_paint_hint = QLabel(
            "Paint the markers in the 2D side views."
        )
        self._split_paint_hint.setWordWrap(True)
        self._split_paint_hint.setStyleSheet(PANEL_HELP_STYLE)
        _s3.addWidget(self._split_paint_hint)
        # It is advice for the EMPTY state.  Once there are markers on
        # screen the list says the same thing better, and the line is a
        # row of the height the list needs.

        _s3 = _s3_outer
        _s3.addWidget(self._split_paint_rows_container)

        layout.addWidget(self._split_s3_container)

        # -- reveal wiring (pure view logic, no callbacks) -----------------
        self.split_guided_summary_btn.toggled.connect(
            self._on_split_sources_toggled
        )
        self.split_advanced_btn = QPushButton("Advanced")
        self.split_seeds_from_combo.currentIndexChanged.connect(
            self._on_split_seeds_from_changed
        )
        self.split_walls_from_combo.currentIndexChanged.connect(
            self._on_split_walls_from_changed
        )
        self.split_seed_detector_combo.currentIndexChanged.connect(
            self._on_split_seed_detector_changed
        )
        self.split_solver_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_item_help("split_solver_combo")
        )
        for _attr, _box_attr in self._SPLIT_SUBSECTIONS:
            getattr(self, _attr).toggled.connect(
                lambda checked, a=_attr:
                self._on_split_subsection_toggled(a, checked)
            )
        for _attr in (
            "split_seed_prep_btn", "split_wall_prep_btn",
        ):
            getattr(self, _attr).toggled.connect(
                lambda checked, a=_attr:
                self._on_split_prep_toggled(a, checked)
            )
        self.split_marker_list.itemChanged.connect(
            self._on_split_marker_check_changed
        )
        # NOT `currentRowChanged`.  The retired
        # `split_seed_list` used it, which was harmless on a list you
        # click rarely -- but this list is TICKED constantly, and Qt
        # moves the current row on a checkbox click too, so every tick
        # would have run `sync_selected_label` across the main viewer
        # and all three subviewers.  `itemClicked` fires once per real
        # click and is suppressed for the click that toggled the tick.
        self.split_marker_list.itemClicked.connect(
            self._on_marker_row_clicked
        )
        self.split_marker_all_btn.clicked.connect(
            lambda: self._set_all_marker_checks(True)
        )
        self.split_marker_none_btn.clicked.connect(
            lambda: self._set_all_marker_checks(False)
        )

        # ==================================================================
        # S4 PREVIEW -- the button, the Stop beside it, the bar underneath.
        #
        # PINNED, not scrolled (2026-09-19).  It used to be the next widget
        # in the scroll content, directly below S3 -- and S3 is the one
        # section on this page whose height the user controls.  Measured on
        # the live dock: with Sources > Markers open on an image source, S3
        # alone is 276-285 px in a 207 px viewport, so Preview sat at y=395
        # with its bottom at 422, off the bottom of a 207 px window.  The
        # driver caught it as 12b/12d; the offscreen test did not, because
        # its fixtures had Sources COLLAPSED, which is not a state the
        # driver (or a user about to choose a marker source) is ever in.
        #
        # Trimming could not fix it: one open sub-section is already bigger
        # than the viewport.  A fixed-height action that is needed in every
        # single state does not belong in a scrolling region, so it now sits
        # between the scroll area and the footer, where nothing can push it.
        # ==================================================================
        self._split_s4_container = QWidget(page)
        _s4 = QVBoxLayout(self._split_s4_container)
        _s4.setContentsMargins(0, 0, 0, 0)
        _s4.setSpacing(2)

        preview_row = QWidget()
        _pr = QHBoxLayout(preview_row)
        _pr.setContentsMargins(0, 0, 0, 0)
        _pr.setSpacing(4)
        self.split_preview_btn = QPushButton("Preview")
        self.split_preview_btn.setStyleSheet(PANEL_PRIMARY_BTN_STYLE)
        self.split_preview_btn.setEnabled(False)
        self.split_preview_btn.setToolTip(
            "Run the split with the chosen solver and show the result. "
            "Nothing is written until you Accept."
        )
        _pr.addWidget(self.split_preview_btn, 1)

        # "Stop waiting" now sits BESIDE the button that starts the work it
        # stops; it used to be five rows above it (finding 8).
        self.split_cancel_preview_btn = QPushButton("Stop waiting")
        self.split_cancel_preview_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_cancel_preview_btn.setToolTip(
            "Give up on this preview. Nothing is changed. The calculation "
            "may take another moment to stop in the background -- its answer "
            "is thrown away."
        )
        self.split_cancel_preview_btn.setVisible(False)
        _pr.addWidget(self.split_cancel_preview_btn)
        _s4.addWidget(preview_row)

        self._split_busy_container = QWidget()
        _busy = QHBoxLayout(self._split_busy_container)
        _busy.setContentsMargins(0, 0, 0, 0)
        _busy.setSpacing(4)
        self.split_busy_bar = QProgressBar()
        self.split_busy_bar.setRange(0, 0)          # marquee until sized
        self.split_busy_bar.setTextVisible(False)
        # `.bar` -- 4 px, gold, no chrome.
        self.split_busy_bar.setFixedHeight(4)
        self.split_busy_bar.setStyleSheet(_PANEL_BUSY_BAR_STYLE)
        _busy.addWidget(self.split_busy_bar, 1)
        self._split_busy_container.setVisible(False)
        _s4.addWidget(self._split_busy_container)

        # Reserve the bar's slot so the strip does not move when a preview
        # starts.  The footer learned this the hard way: a pinned thing
        # that changes height is not pinned, and the button the user has
        # just clicked is the last one that should jump under the cursor.
        self._split_s4_container.setMinimumHeight(
            preview_row.sizeHint().height() + _s4.spacing()
            + self.split_busy_bar.height()
        )

        # Into the PAGE, between the scroll area and the footer -- NOT into
        # `layout`, which is the scroll's content.
        page.layout().insertWidget(
            page.layout().indexOf(footer), self._split_s4_container, 0
        )

        # ==================================================================
        # S5 RESULT -- ABOVE the footer.  These rows used to be appended
        # BELOW the Accept/Reject buttons that act on them (finding 2).
        # ==================================================================
        self._split_result_container = QWidget()
        self._split_result_container.setVisible(False)
        _s5 = QVBoxLayout(self._split_result_container)
        # The mockup's RESULT `.sec`: one titled card holding the warning,
        # the pieces and the line that says what an unticked one means.
        make_panel_card(self._split_result_container)

        # Brick U3 -- "Show what the tool sees", ON THE
        # CARD HEADER ROW.
        #
        # The card title used to be a bare label here.  The tick costs a
        # row of its own 15 px, and review-with-markers-open had 6 px of
        # its 753 px ceiling left after Brick U4 -- a number that
        # may not be raised again without the owner.  Sharing the header
        # row costs the page NOTHING while the box is off, because the
        # row is already there and a checkbox is shorter than the title.
        # (The alternative, the S2 object card, was measured and does not
        # fit sideways either: the two tick labels want 463 px of a
        # 321 px card.)
        _s5_head = QHBoxLayout()
        _s5_head.setContentsMargins(0, 0, 0, 0)
        _s5_head.setSpacing(6)
        _s5_head.addWidget(
            panel_card_header("Result", self._split_result_container))
        _s5_head.addStretch(1)
        self.split_show_walls_cb = QCheckBox(
            "Show what the tool sees", self._split_result_container)
        self.split_show_walls_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        # Disabled at birth and re-disabled at every exit: there is no
        # elevation to show until an engine Preview has produced one, and
        # S5 itself is hidden outside review, so the control is never
        # reachable when it would mean nothing.
        self.split_show_walls_cb.setEnabled(False)
        self.split_show_walls_cb.setToolTip(WALLS_TOGGLE_DISABLED)
        self.split_show_walls_cb.toggled.connect(
            self._on_split_show_walls_toggled)
        _s5_head.addWidget(self.split_show_walls_cb, 0)
        _s5.addLayout(_s5_head)

        # ...and the legend, which is on screen ONLY while the box is
        # ticked, so it costs nothing unless the user opted in.
        self.split_walls_legend = QLabel("", self._split_result_container)
        self.split_walls_legend.setWordWrap(True)
        self.split_walls_legend.setStyleSheet(PANEL_HELP_STYLE)
        self.split_walls_legend.setVisible(False)
        _s5.addWidget(self.split_walls_legend)

        # The PERMANENT half of the pair.  Since the loop
        # landed, the marker list and the piece list are on screen at the
        # same time -- so each card says, in one line that never changes,
        # which of the two it is.  `split_piece_caption` below is the
        # per-cut sentence and is overwritten; this one is not.
        self.split_piece_card_caption = QLabel(
            "Pieces - after the cut. Decides what Accept writes."
        )
        # Elided like its twin: review-with-markers-open is the tallest
        # state the panel has, and a caption that wraps to two lines
        # costs it a row of pieces.
        self.split_piece_card_caption.setWordWrap(False)
        self.split_piece_card_caption.setToolTip(
            "Pieces - after the cut. Decides what Accept writes."
        )
        self.split_piece_card_caption.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.split_piece_card_caption.setStyleSheet(PANEL_HELP_STYLE)
        _s5.addWidget(self.split_piece_card_caption)

        # ------------------------------------------------------------------
        # Guided Split -- the checkable piece list.
        #
        # Unticking a piece does NOT re-solve; it only narrows what Accept
        # commits.  On a split, an unticked piece RETURNS TO THE TARGET CELL
        # -- it is never deleted tissue, and the caption
        # says so in words.
        # ------------------------------------------------------------------
        # The caption's wording, shipped now so the guide is not
        # describing a caption that does not exist.  The orchestrator
        # still overwrites it with the cell-specific line at review.
        # The caption sits BELOW the list now (the mockup's RESULT card),
        # so it can no longer introduce it -- it says what an unticked
        # piece MEANS, which is what a line under a list is for.  The
        # orchestrator still overwrites it with the cell-specific wording
        # at review.
        self.split_piece_caption = QLabel(
            "Unticked pieces stay part of the original object. "
            "Nothing is deleted."
        )
        self.split_piece_caption.setWordWrap(True)
        self.split_piece_caption.setStyleSheet(PANEL_HELP_STYLE)
        self.split_piece_caption.setVisible(False)

        self.split_piece_list = QListWidget()
        self.split_piece_list.setMaximumHeight(_PIECE_LIST_MAX_H)
        self._style_panel_list(self.split_piece_list)
        self.split_piece_list.setToolTip(
            "The pieces this cut would make. Tick the ones you want as "
            "separate objects; anything you untick stays part of the original "
            "object."
        )
        self.split_piece_list.itemChanged.connect(
            self._on_split_piece_check_changed
        )
        self.split_piece_list.setVisible(False)

        # The quality gate's sentences.  One label, one line
        # per note, ABOVE everything else in the card: they are what the
        # user has to read before deciding what to do with the pieces.
        self.split_quality_label = QLabel("")
        self.split_quality_label.setWordWrap(True)
        self.split_quality_label.setStyleSheet(PANEL_WARN_STYLE)
        self.split_quality_label.setVisible(False)

        self.split_arbiter_msg_label = QLabel("")
        self.split_arbiter_msg_label.setWordWrap(True)
        self.split_arbiter_msg_label.setStyleSheet(PANEL_WARN_STYLE)
        self.split_arbiter_msg_label.setVisible(False)

        # Screen order, per the mockup's RESULT card: what went wrong,
        # then the pieces, then what an unticked one means.  The list used
        # to sit above the warning that explains it.
        _s5.addWidget(self.split_quality_label)
        _s5.addWidget(self.split_arbiter_msg_label)
        _s5.addWidget(self._list_with_grip(
            self.split_piece_list, "split_piece_list",
            _PIECE_LIST_H_PROP, _PIECE_LIST_MAX_H))
        _s5.addWidget(self.split_piece_caption)

        self.split_anyway_btn = QPushButton("Split anyway")
        self.split_anyway_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_anyway_btn.setToolTip(
            "Make the cut the tool declined to make. The result is marked so "
            "that later you can still tell it apart from a cut the wall "
            "signal justified."
        )
        self.split_anyway_btn.setVisible(False)
        _s5.addWidget(self.split_anyway_btn)

        # --- Fragment Refine section ---
        self.split_refine_further_btn = QPushButton("Refine Fragments...")
        self.split_refine_further_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_refine_further_btn.setToolTip(
            "Write the pieces into the volume and step through them HERE, "
            "one at a time -- it does not switch tabs. Each piece can be "
            "smoothed, dilated or grown on its own; Accept all commits every "
            "one of them, Discard puts the original object back. Offered "
            "after every Preview, whichever way the markers and walls were "
            "chosen; an unticked piece stays part of the original object."
        )
        self.split_refine_further_btn.setEnabled(False)
        # S5 is empty until there is a result to talk about, so the hand-off
        # button is hidden rather than greyed until the pieces exist.
        self.split_refine_further_btn.setVisible(False)
        _s5.addWidget(self.split_refine_further_btn)

        self.split_fragment_list = QListWidget()
        self.split_fragment_list.setMaximumHeight(120)
        self._style_panel_list(self.split_fragment_list)
        self.split_fragment_list.setToolTip(
            "Click a fragment to select it for refinement"
        )
        self.split_fragment_list.setVisible(False)
        _s5.addWidget(self.split_fragment_list)

        self.split_engine_msg_label = QLabel("")
        self.split_engine_msg_label.setWordWrap(True)
        self.split_engine_msg_label.setStyleSheet(PANEL_HELP_STYLE)
        self.split_engine_msg_label.setVisible(False)
        _s5.addWidget(self.split_engine_msg_label)

        # A refusal is not one of the engine's remarks (slot S5:
        # "the engine message (refusals red)").  It gets its own label,
        # in red, because it is the absence of a result rather than a
        # warning about one -- and it was being rendered in the same
        # grey as the "Wall contrast" line directly under it.
        self.split_engine_error_label = QLabel("")
        self.split_engine_error_label.setWordWrap(True)
        self.split_engine_error_label.setStyleSheet(PANEL_ERROR_STYLE)
        self.split_engine_error_label.setVisible(False)
        _s5.addWidget(self.split_engine_error_label)

        layout.addWidget(self._split_result_container)

        # ==================================================================
        # SA ADVANCED -- one disclosure deep, and the LAST thing inside the
        # scroll content so opening it just makes the page longer.  Grouped into the mockup's three `.sec` cards (MARKERS /
        # WALLS / CUT AND SNAP) over ONE fixed label column: a QFormLayout
        # gave every row its own left edge and, at 486 px of minimum width
        # in a 329 px viewport, CLIPPED the page's right edge.
        # ==================================================================
        self.split_advanced_btn.setCheckable(True)
        self.split_advanced_btn.setChecked(False)
        self.split_advanced_btn.setStyleSheet(PANEL_CHEVRON_STYLE)
        self.split_advanced_btn.setToolTip(
            "Everything else the engine can do. All of it is visible at "
            "once -- nothing is hidden behind another menu."
        )
        layout.addWidget(self.split_advanced_btn)

        self._split_advanced_container = QWidget()
        _adv_col = QVBoxLayout(self._split_advanced_container)
        _adv_col.setContentsMargins(0, 0, 0, 0)
        _adv_col.setSpacing(2)

        def _adv_group(title):
            """One `.sec` card with an UPPERCASE header; returns its layout."""
            box = QWidget(self._split_advanced_container)
            box_lay = QVBoxLayout(box)
            make_panel_card(box)
            box_lay.addWidget(panel_card_header(title, self))
            _adv_col.addWidget(box)
            return box_lay

        def _adv_row(box_lay, text, field, *, indent=0, attr=None):
            """A knob on the shared label column (the mockup's `.row`),
            with its grey sentence under it.

            `attr` is the widget's attribute name -- the key HELP_LINES
            is held to set equality against, so a knob added without one
            fails `test_helper_lines` rather than shipping unexplained.
            """
            row = QHBoxLayout()
            row.setContentsMargins(indent, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(panel_form_label(
                text, PANEL_ADV_LABEL_COLUMN - indent, self))
            row.addWidget(field, 1)
            box_lay.addLayout(row)
            if attr and attr in HELP_LINES:
                line = self._helper_line(attr)
                line.setContentsMargins(indent, 0, 0, 0)
                box_lay.addWidget(line)

        # ---- MARKERS ---------------------------------------------------
        _markers = _adv_group("Markers")

        self.split_seed_h_sb = QDoubleSpinBox()
        self.split_seed_h_sb.setRange(0.001, 1.0)
        self.split_seed_h_sb.setSingleStep(0.01)
        self.split_seed_h_sb.setDecimals(3)
        self.split_seed_h_sb.setValue(0.15)
        self.split_seed_h_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_h_sb.setToolTip(
            "One number, two jobs. For the peak and interior finders it is "
            "how far a peak must stand out from what is around it (for the "
            "interiors, how deep the basin must be) -- lower it if markers "
            "are missing.\n\n"
            "For the two blob finders it is the detection threshold "
            "instead, and it works the same way round: lower finds more "
            "blobs."
        )
        _adv_row(_markers, "Prominence", self.split_seed_h_sb,
                 attr="split_seed_h_sb")

        # "The single most useful knob" -- the sandbox Tips cell.  It was
        # the one engine parameter with no panel row at all.
        self.split_seed_smooth_sb = QDoubleSpinBox()
        self.split_seed_smooth_sb.setRange(0.0, 20.0)
        self.split_seed_smooth_sb.setSingleStep(0.5)
        self.split_seed_smooth_sb.setDecimals(2)
        self.split_seed_smooth_sb.setValue(2.0)
        self.split_seed_smooth_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_smooth_sb.setToolTip(
            "How much the marker channel is smoothed before peaks are "
            "found. More smoothing gives fewer, larger markers."
        )
        _adv_row(_markers, "Marker blur", self.split_seed_smooth_sb,
                 attr="split_seed_smooth_sb")

        self.split_min_distance_sb = QDoubleSpinBox()
        self.split_min_distance_sb.setRange(0.0, 100.0)
        self.split_min_distance_sb.setValue(5.0)
        self.split_min_distance_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_min_distance_sb.setToolTip(
            "Closest two markers may be, in microns. Converted using the "
            "FINEST voxel spacing, because the underlying peak finder takes "
            "one number for all three axes."
        )
        _adv_row(_markers, "Min marker gap (um)", self.split_min_distance_sb,
                 attr="split_min_distance_sb")

        self.split_max_sigma_sb = QDoubleSpinBox()
        self.split_max_sigma_sb.setRange(0.5, 100.0)
        self.split_max_sigma_sb.setSingleStep(0.5)
        self.split_max_sigma_sb.setDecimals(2)
        self.split_max_sigma_sb.setValue(6.0)
        self.split_max_sigma_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_max_sigma_sb.setToolTip(
            "The blob finders only: blobs bigger across than this are "
            "ignored, in microns."
        )
        _adv_row(_markers, "Largest nucleus (um)", self.split_max_sigma_sb,
                 attr="split_max_sigma_sb")

        self.split_seed_thresh_sb = QDoubleSpinBox()
        self.split_seed_thresh_sb.setRange(0.0, 1.0)
        self.split_seed_thresh_sb.setSingleStep(0.05)
        self.split_seed_thresh_sb.setDecimals(2)
        self.split_seed_thresh_sb.setValue(0.0)
        # The minimum IS "leave it to the tool": a blank spin box cannot
        # be typed into, and a separate tick for one knob is a row the
        # 247 px viewport does not have.
        self.split_seed_thresh_sb.setSpecialValueText("automatic")
        self.split_seed_thresh_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_thresh_sb.setToolTip(
            "Nucleus peaks and Bright-object centres: how bright a nucleus "
            "must be. Leave it on automatic and the tool picks the level "
            "itself."
        )
        _adv_row(_markers, "Marker threshold", self.split_seed_thresh_sb,
                 attr="split_seed_thresh_sb")

        self.split_seed_clip_sb = QDoubleSpinBox()
        self.split_seed_clip_sb.setRange(0.0, 0.2)
        self.split_seed_clip_sb.setSingleStep(0.005)
        self.split_seed_clip_sb.setDecimals(3)
        self.split_seed_clip_sb.setValue(0.02)
        self.split_seed_clip_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_clip_sb.setToolTip(
            "Local contrast on the marker channel. Higher gives more "
            "contrast, and lets more noise through with it."
        )
        _adv_row(_markers, "Local contrast strength",
                 self.split_seed_clip_sb, attr="split_seed_clip_sb")

        self.split_lift_mode_combo = QComboBox()
        self.split_lift_mode_combo.addItems(["gentle", "strong"])
        self.split_lift_mode_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_lift_mode_combo.setToolTip(
            "Which kind of lift the 'Lift dim signal' tick runs: gentle "
            "raises faint signal a little, strong raises it a lot."
        )
        _adv_row(_markers, "Lift dim signal", self.split_lift_mode_combo,
                 attr="split_lift_mode_combo")

        self.split_seed_gamma_sb = QDoubleSpinBox()
        self.split_seed_gamma_sb.setRange(0.05, 2.0)
        self.split_seed_gamma_sb.setSingleStep(0.05)
        self.split_seed_gamma_sb.setDecimals(2)
        self.split_seed_gamma_sb.setValue(0.5)
        self.split_seed_gamma_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_gamma_sb.setToolTip(
            "How hard the gentle lift pulls: smaller numbers lift faint "
            "signal more."
        )
        _adv_row(_markers, "Lift amount", self.split_seed_gamma_sb,
                 indent=_SUB_INDENT, attr="split_seed_gamma_sb")

        self.split_seed_bg_sb = QSpinBox()
        self.split_seed_bg_sb.setRange(1, 200)
        self.split_seed_bg_sb.setValue(15)
        self.split_seed_bg_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_bg_sb.setToolTip(
            "'Remove haze': the largest feature to keep, in voxels. Set it "
            "bigger than a nucleus or it will eat the nuclei too."
        )
        _adv_row(_markers, "Haze size", self.split_seed_bg_sb,
                 attr="split_seed_bg_sb")

        self.split_seed_denoise_sb = QSpinBox()
        self.split_seed_denoise_sb.setRange(1, 10)
        self.split_seed_denoise_sb.setValue(2)
        self.split_seed_denoise_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_seed_denoise_sb.setToolTip(
            "'Despeckle': how many voxels around each point are averaged."
        )
        _adv_row(_markers, "Despeckle size", self.split_seed_denoise_sb,
                 attr="split_seed_denoise_sb")

        # ---- WALLS -----------------------------------------------------
        _walls = _adv_group("Walls")

        self.split_wall_clip_sb = QDoubleSpinBox()
        self.split_wall_clip_sb.setRange(0.0, 0.2)
        self.split_wall_clip_sb.setSingleStep(0.005)
        self.split_wall_clip_sb.setDecimals(3)
        self.split_wall_clip_sb.setValue(0.01)
        self.split_wall_clip_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_wall_clip_sb.setToolTip(
            "How much of the brightest and dimmest signal to ignore before "
            "the walls are read out of the channel. Larger values tolerate "
            "hot spots."
        )
        _adv_row(_walls, "Wall clip", self.split_wall_clip_sb,
                 attr="split_wall_clip_sb")

        self.split_wall_smooth_sb = QDoubleSpinBox()
        self.split_wall_smooth_sb.setRange(0.0, 20.0)
        self.split_wall_smooth_sb.setSingleStep(0.5)
        self.split_wall_smooth_sb.setDecimals(2)
        self.split_wall_smooth_sb.setValue(1.0)
        self.split_wall_smooth_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_wall_smooth_sb.setToolTip(
            "Smooths speckle in the wall channel before the ridge filter "
            "looks for sheets in it."
        )
        _adv_row(_walls, "Wall blur", self.split_wall_smooth_sb,
                 attr="split_wall_smooth_sb")

        self.split_arbiter_pct_sb = QDoubleSpinBox()
        self.split_arbiter_pct_sb.setRange(1.0, 99.0)
        self.split_arbiter_pct_sb.setValue(75.0)
        self.split_arbiter_pct_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_arbiter_pct_sb.setToolTip(
            "How strong the wall has to be along a proposed cut before "
            "the cut is kept, measured against the rest of THIS object's own "
            "wall signal rather than an absolute level. 75 means stronger "
            "than three-quarters of it; higher demands more evidence."
        )
        _adv_row(_walls, "Wall required (%)", self.split_arbiter_pct_sb,
                 attr="split_arbiter_pct_sb")

        self.split_wall_ridge_cb = QCheckBox("Sheet-ridge filter")
        self.split_wall_ridge_cb.setChecked(True)
        self.split_wall_ridge_cb.setToolTip(
            "Look for thin bright sheets in the wall channel rather than "
            "raw brightness. Leave it on for a stain that marks where "
            "objects meet; turn it off to follow brightness directly."
        )
        _walls.addWidget(self.split_wall_ridge_cb)
        _walls.addWidget(self._helper_line("split_wall_ridge_cb"))

        self.split_search_padding_sb = QSpinBox()
        self.split_search_padding_sb.setRange(0, 30)
        self.split_search_padding_sb.setValue(15)
        self.split_search_padding_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_search_padding_sb.setToolTip(
            "How far outside the object the tool looks, in voxels.  Read "
            "when you press Start -- changing it mid-session does nothing "
            "until the next object."
        )
        _adv_row(_walls, "Search padding (voxels)",
                 self.split_search_padding_sb,
                 attr="split_search_padding_sb")

        # ---- CUT AND SNAP ----------------------------------------------
        _cut = _adv_group("Cut and snap")

        self.split_rw_beta_sb = QDoubleSpinBox()
        self.split_rw_beta_sb.setRange(1.0, 1000.0)
        self.split_rw_beta_sb.setValue(130.0)
        self.split_rw_beta_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_rw_beta_sb.setToolTip(
            "Soft cut only: how strongly a wall resists the cut. Higher "
            "makes the cut stick more tightly to the walls it finds."
        )
        _adv_row(_cut, "Soft cut: wall resistance", self.split_rw_beta_sb,
                 attr="split_rw_beta_sb")

        self.split_compactness_sb = QDoubleSpinBox()
        # 0-1, not 0-1000 (parity row 47).  The sandbox's own range was
        # 0-0.2; anything past 1 is a number with no behaviour behind it,
        # and it made every step of the spin box useless.
        self.split_compactness_sb.setRange(0.0, 1.0)
        self.split_compactness_sb.setSingleStep(0.05)
        self.split_compactness_sb.setDecimals(3)
        self.split_compactness_sb.setValue(0.0)
        self.split_compactness_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_compactness_sb.setToolTip(
            "Sharp cut only: higher values push the pieces toward equal "
            "size, which is the fix for a starved sliver."
        )
        _adv_row(_cut, "Balance", self.split_compactness_sb,
                 attr="split_compactness_sb")

        self.split_refine_radius_sb = QSpinBox()
        self.split_refine_radius_sb.setRange(1, 30)
        self.split_refine_radius_sb.setValue(4)
        self.split_refine_radius_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_refine_radius_sb.setToolTip(
            "'Snap to the wall signal' only: how far an edge may slide, "
            "in voxels."
        )
        _adv_row(_cut, "Snap radius", self.split_refine_radius_sb,
                 attr="split_refine_radius_sb")

        self.split_refine_eps_sb = QDoubleSpinBox()
        self.split_refine_eps_sb.setRange(0.0001, 1.0)
        self.split_refine_eps_sb.setSingleStep(0.005)
        self.split_refine_eps_sb.setDecimals(4)
        self.split_refine_eps_sb.setValue(0.01)
        self.split_refine_eps_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_refine_eps_sb.setToolTip(
            "'Snap to the wall signal' only: lower numbers snap harder to "
            "the edge in the image."
        )
        _adv_row(_cut, "Snap strength", self.split_refine_eps_sb,
                 attr="split_refine_eps_sb")

        self.split_refine_merge_small_cb = QCheckBox("Absorb slivers below")
        # ON, at 20 -- the sandbox ALWAYS applied MIN_SIZE=20 and the
        # owner asked for that behaviour back.  This is the ONE
        # declared exception to the golden dict's rule 7, and
        # `test_split_params_golden` names it.
        self.split_refine_merge_small_cb.setChecked(True)
        self.split_refine_merge_small_cb.setToolTip(
            "Fold pieces below the size below into their biggest neighbour. "
            "A piece backed by its own marker is never absorbed."
        )
        _cut.addWidget(self.split_refine_merge_small_cb)
        _cut.addWidget(self._helper_line("split_refine_merge_small_cb"))

        self.split_min_size_sb = QSpinBox()
        self.split_min_size_sb.setRange(0, 100000)
        self.split_min_size_sb.setValue(20)
        self.split_min_size_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_min_size_sb.setToolTip("Sliver size, in voxels.")
        _adv_row(_cut, "Sliver size", self.split_min_size_sb,
                 indent=_SUB_INDENT, attr="split_min_size_sb")

        # Smoothing of the cut pieces -- read by the manual path at
        # preview time, so it stays in Advanced rather than disappearing.
        self.split_smooth_mode_combo = QComboBox()
        self.split_smooth_mode_combo.addItems(list(SMOOTH_MODE_LABELS))
        self.split_smooth_mode_combo.setCurrentIndex(SMOOTH_MODE_PIECES)
        self.split_smooth_mode_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_smooth_mode_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.split_smooth_mode_combo.setMinimumContentsLength(10)
        self.split_smooth_mode_combo.setToolTip(
            "Rounds off the staircase edges a cut leaves behind.\n\n"
            "Smooth pieces: each piece is smoothed on its own. Thin pieces "
            "can shrink or vanish, and the object can lose voxels.\n"
            "Keep the object: the cuts between pieces can move, but never "
            "across a painted marker. The object keeps every voxel it "
            "had.\n\n"
            "Two things it is easy to get wrong about this choice: it runs at "
            "PREVIEW, so what you see is already smoothed and what you see "
            "is what Accept writes; and it reaches the HAND-PAINTED path "
            "only. A cut that came from 'Markers from' or from any wall "
            "other than 'Pinch points' ignores it."
        )
        _adv_row(_cut, "Smoothing", self.split_smooth_mode_combo,
                 attr="split_smooth_mode_combo")
        # The old tick, kept as an adapter for code that still asks
        # "is smoothing on?".  Never shown; the combo drives it.
        self.split_auto_smooth_cb = QCheckBox("Auto-smooth results", self)
        self.split_auto_smooth_cb.setChecked(True)
        self.split_auto_smooth_cb.setVisible(False)
        self.split_auto_smooth_cb.setToolTip(
            "Kept for older code; the Smoothing choice above drives it."
        )

        # Auto-smooth sigma container (hidden when auto-smooth unchecked)
        self._smooth_sigma_container = QWidget()
        smooth_sigma_row = QHBoxLayout(self._smooth_sigma_container)
        smooth_sigma_row.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        smooth_sigma_row.setSpacing(4)
        smooth_sigma_row.addWidget(panel_form_label(
            _SIGMA_Z_LABEL, _SIGMA_LABEL_COLUMN, self))
        self.smooth_sigma_z_sb = QDoubleSpinBox()
        self.smooth_sigma_z_sb.setRange(0.1, 10.0)
        self.smooth_sigma_z_sb.setSingleStep(0.5)
        self.smooth_sigma_z_sb.setDecimals(1)
        self.smooth_sigma_z_sb.setValue(3.0)
        self.smooth_sigma_z_sb.setToolTip(
            "Smoothing strength along Z (depth). Higher = smoother between slices. "
            "Default 3.0 works for most data."
        )
        smooth_sigma_row.addWidget(self.smooth_sigma_z_sb, 1)
        smooth_sigma_row.addWidget(panel_form_label(
            _SIGMA_XY_LABEL, _SIGMA_LABEL_COLUMN, self))
        self.smooth_sigma_xy_sb = QDoubleSpinBox()
        self.smooth_sigma_xy_sb.setRange(0.1, 10.0)
        self.smooth_sigma_xy_sb.setSingleStep(0.1)
        self.smooth_sigma_xy_sb.setDecimals(1)
        self.smooth_sigma_xy_sb.setValue(1.5)
        self.smooth_sigma_xy_sb.setToolTip(
            "Smoothing strength in XY plane. Higher = rounder cross-sections. "
            "Default 1.5 is conservative."
        )
        smooth_sigma_row.addWidget(self.smooth_sigma_xy_sb, 1)
        _cut.addWidget(self._smooth_sigma_container)
        # Wire show/hide to auto-smooth checkbox
        self.split_smooth_mode_combo.currentIndexChanged.connect(
            self._on_split_smooth_mode_changed
        )

        # Auto-advance after accept toggle.
        self.split_auto_advance_cb = QCheckBox("Auto-advance after accept")
        self.split_auto_advance_cb.setChecked(True)
        self.split_auto_advance_cb.setToolTip(
            "Advance to the next triage label after accepting a split"
        )
        _cut.addWidget(self.split_auto_advance_cb)
        _cut.addWidget(self._helper_line("split_auto_advance_cb"))

        # How many new pieces stay isolated after Accept.  Each piece costs
        # a small 3D layer, so a large cut shows only its largest piece.  Read
        # when Accept is pressed; not saved between sessions.
        self.split_pieces_cap_sb = QSpinBox()
        self.split_pieces_cap_sb.setRange(1, 50)
        self.split_pieces_cap_sb.setValue(10)
        self.split_pieces_cap_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.split_pieces_cap_sb.setToolTip(
            "After Accept, up to this many of the new pieces stay isolated "
            "in the 3D view.  With more pieces than this, only the largest "
            "one stays."
        )
        _adv_row(_cut, "Pieces kept in view after Accept",
                 self.split_pieces_cap_sb, attr="split_pieces_cap_sb")

        self._split_advanced_container.setVisible(False)
        layout.addWidget(self._split_advanced_container)
        self.split_advanced_btn.toggled.connect(
            self._on_split_advanced_toggled
        )
        self.split_advanced_btn.setText(
            "{} Advanced".format(_CHEVRON_CLOSED)
        )

        layout.addStretch(1)

        # ==================================================================
        # S6 FOOTER -- two buttons and one line, and nothing else.
        # ==================================================================
        _ft = QVBoxLayout(footer)
        _ft.setContentsMargins(7, 5, 7, 4)
        _ft.setSpacing(3)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.split_accept_btn = QPushButton("Accept")
        # `.btn.pri.grow` -- the one committing action, solid gold and
        # taking the slack; the mockup gives Discard only its own words.
        self.split_accept_btn.setStyleSheet(PANEL_PRIMARY_BTN_STYLE)
        self.split_accept_btn.setEnabled(False)
        self.split_accept_btn.setToolTip(
            "Write the ticked pieces into the volume as separate objects. "
            "Anything you unticked stays part of the original object -- it is "
            "never deleted."
        )
        # Born as "Cancel": nothing pushes an idle state when the dock
        # opens, so the constructor's text is what the user reads first --
        # and at birth nothing is staged.
        self.split_reject_btn = QPushButton("Cancel")
        self.split_reject_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.split_reject_btn.setEnabled(False)
        self.split_reject_btn.setToolTip(
            "Put the object back exactly as it was and end this session. "
            "From Start until you press this, the object is held out of the "
            "volume, which is why it is hidden in the views."
        )
        btn_row.addWidget(self.split_accept_btn, 1)
        btn_row.addWidget(self.split_reject_btn, 0)
        _ft.addLayout(btn_row)

        # Filled from the secondary's text at the end of this builder, so
        # the line and the button cannot disagree at birth.
        self._split_footer_summary = self._footer_summary_label()
        _ft.addWidget(self._split_footer_summary)

        # Markers opens first: it is the sub-section the user answers
        # before anything else, and one of the three has to be open or
        # S3 is a row of three closed buttons.
        self.split_markers_btn.setChecked(True)
        self._on_split_seeds_from_changed(
            self.split_seeds_from_combo.currentIndex()
        )
        self._on_split_walls_from_changed(
            self.split_walls_from_combo.currentIndex()
        )
        self._refresh_item_help("split_seed_detector_combo")
        self._refresh_item_help("split_solver_combo")
        # Two of the marker knobs (prominence, marker gap) live in
        # ADVANCED, which is built AFTER S3 -- wiring them up there
        # would have wired nothing at all.
        self._wire_marker_knob_invalidation()
        self._refresh_split_prep_chips()
        self._refresh_marker_count_line()
        self._apply_split_sources_visibility()
        self._refresh_split_guided_summary()
        self._refresh_split_footer_summary()
        self._register_page(
            "Reshape", page, scroll, content, footer,
            status=self.split_status_label, preview=self.split_preview_btn,
        )
        return page

    # --- Assisted Split callback / state API --------------------------------

    def set_split_callbacks(
        self,
        *,
        on_start=None,
        on_next_seed=None,
        on_undo_seed=None,
        on_refine_seeds=None,
        on_select_seed=None,
        on_preview=None,
        on_accept=None,
        on_reject=None,
        on_edt_toggle=None,
        on_refine_further=None,
        on_select_fragment=None,
        on_detect_seeds=None,
        on_threshold_changed=None,
        # --- the guided-Split additions ---
        on_seeds_from_changed=None,
        on_walls_from_changed=None,
        on_guide_browse=None,
        on_guide_load=None,
        on_guide_show_toggled=None,
        on_piece_check_changed=None,
        on_cancel_preview=None,
        on_split_anyway=None,
        # --- Brick 2a: the marker picker ---
        on_find_markers=None,
        on_markers_invalidated=None,
        # --- Brick U1: the loop ---
        on_cut_knob_changed=None,
        on_marker_check_changed=None,
        # --- Brick U2: the outline tick ---
        on_contour_toggled=None,
        # --- Brick U3: "Show what the tool sees" ---
        on_show_walls_toggled=None,
    ):
        """Wire button clicks to external orchestration callbacks.

        Parameters
        ----------
        on_start : callable() or None
        on_next_seed : callable() or None
        on_undo_seed : callable() or None
        on_refine_seeds : callable() or None
        on_select_seed : callable(int) or None
        on_preview : callable() or None
        on_accept : callable() or None
        on_reject : callable() or None
        on_edt_toggle : callable(bool) or None
        on_refine_further : callable() or None
        on_select_fragment : callable(int) or None
        on_detect_seeds : callable() or None
        on_threshold_changed : callable(float) or None
        """
        if on_start and hasattr(self, "split_start_btn"):
            try:
                self.split_start_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_start_btn.clicked.connect(on_start)
        if on_next_seed and hasattr(self, "split_next_seed_btn"):
            try:
                self.split_next_seed_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_next_seed_btn.clicked.connect(on_next_seed)
        if on_undo_seed and hasattr(self, "split_undo_seed_btn"):
            try:
                self.split_undo_seed_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_undo_seed_btn.clicked.connect(on_undo_seed)
        if on_refine_seeds and hasattr(self, "split_refine_seeds_btn"):
            try:
                self.split_refine_seeds_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_refine_seeds_btn.clicked.connect(on_refine_seeds)
        if on_select_seed:
            # Stored only.  The marker list is wired to `itemClicked` in
            # the builder, NOT to `currentRowChanged`:
            # Qt moves the current row on a checkbox click too, so that
            # connection made every tick run a four-viewer
            # `sync_selected_label`.
            self._on_select_seed_cb = on_select_seed
        if on_preview and hasattr(self, "split_preview_btn"):
            try:
                self.split_preview_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_preview_btn.clicked.connect(on_preview)
        if on_accept and hasattr(self, "split_accept_btn"):
            try:
                self.split_accept_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_accept_btn.clicked.connect(on_accept)
        if on_reject and hasattr(self, "split_reject_btn"):
            try:
                self.split_reject_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_reject_btn.clicked.connect(on_reject)
        if on_edt_toggle and hasattr(self, "split_edt_cb"):
            try:
                self.split_edt_cb.toggled.disconnect()
            except TypeError:
                pass
            self.split_edt_cb.toggled.connect(on_edt_toggle)
        if on_detect_seeds and hasattr(self, "split_detect_seeds_btn"):
            try:
                self.split_detect_seeds_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_detect_seeds_btn.clicked.connect(on_detect_seeds)
        if on_threshold_changed is not None:
            self._on_threshold_changed_cb = on_threshold_changed
        if on_refine_further and hasattr(self, "split_refine_further_btn"):
            try:
                self.split_refine_further_btn.clicked.disconnect()
            except TypeError:
                pass
            self.split_refine_further_btn.clicked.connect(on_refine_further)
        if on_select_fragment and hasattr(self, "split_fragment_list"):
            self._on_select_fragment_cb = on_select_fragment
            try:
                self.split_fragment_list.currentItemChanged.disconnect()
            except TypeError:
                pass
            self.split_fragment_list.currentItemChanged.connect(
                self._on_fragment_item_changed
            )

        # --- guided-Split wiring ---------------------------
        # Each is stashed so the pure view helpers can fire it, and each
        # button is disconnect-then-connect exactly like the rows above, so
        # re-wiring on relaunch cannot stack duplicate connections.
        if on_seeds_from_changed is not None:
            self._on_seeds_from_changed_cb = on_seeds_from_changed
        if on_walls_from_changed is not None:
            self._on_walls_from_changed_cb = on_walls_from_changed
        if on_piece_check_changed is not None:
            self._on_split_piece_check_cb = on_piece_check_changed
        if on_markers_invalidated is not None:
            # Stashed, not connected: it is fired from the pure view
            # helper that empties the list, so re-wiring on relaunch
            # cannot stack duplicate connections.
            self._on_markers_invalidated_cb = on_markers_invalidated
        # Brick U1: the two that make S3 leave review.  Stashed for the
        # same reason -- both are fired from pure view helpers.
        #
        # `on_marker_check_changed` is no longer review-only:
        # it fires on EVERY marker tick, because an unticked marker now
        # comes off the object and that is just as true with no preview
        # on screen.  Exactly once per tick -- see the blockSignals in
        # `_on_split_marker_check_changed`.
        if on_cut_knob_changed is not None:
            self._on_cut_knob_changed_cb = on_cut_knob_changed
        if on_marker_check_changed is not None:
            self._on_marker_check_changed_cb = on_marker_check_changed
        for _cb, _attr in (
            (on_guide_browse, "split_guide_browse_btn"),
            (on_guide_load, "split_guide_load_btn"),
            (on_cancel_preview, "split_cancel_preview_btn"),
            (on_split_anyway, "split_anyway_btn"),
            (on_find_markers, "split_find_markers_btn"),
        ):
            if _cb and hasattr(self, _attr):
                _btn = getattr(self, _attr)
                try:
                    _btn.clicked.disconnect()
                except TypeError:
                    pass
                _btn.clicked.connect(_cb)
        if on_guide_show_toggled and hasattr(self, "split_guide_show_cb"):
            try:
                self.split_guide_show_cb.toggled.disconnect()
            except TypeError:
                pass
            self.split_guide_show_cb.toggled.connect(on_guide_show_toggled)
        if on_contour_toggled and hasattr(self, "split_contour_cb"):
            try:
                self.split_contour_cb.toggled.disconnect()
            except TypeError:
                pass
            self.split_contour_cb.toggled.connect(on_contour_toggled)
        # Brick U3: STORED, not connected.  The box's own `toggled` is
        # already wired to `_on_split_show_walls_toggled`, which swaps
        # the legend before it forwards -- and `clear_walls_toggle`
        # needs an untick that reaches neither.
        if on_show_walls_toggled:
            self._on_show_walls_toggled_cb = on_show_walls_toggled

    def _on_split_about_toggled(self, checked: bool) -> None:
        """Open / close the page description."""
        self.split_about_btn.setText(
            _CHEVRON_OPEN if checked else _CHEVRON_CLOSED)
        self._refresh_split_about_body()

    def _refresh_split_about_body(self) -> None:
        """The body is on screen iff the chevron is open AND no session.

        ONE predicate, read by both callers -- the chevron's own toggle
        and `update_split_ui_state`'s Start/idle sweep -- so the two can
        never disagree about it.  `_split_is_active` and not
        `isVisibleTo`, because during construction and while the dock is
        closed Qt answers that question about the whole parent chain,
        not about the thing we actually mean.
        """
        body = getattr(self, "_split_about_body", None)
        if body is None:
            return
        body.setVisible(
            bool(self.split_about_btn.isChecked())
            and not bool(getattr(self, "_split_is_active", False))
        )

    def _on_fragment_item_changed(self, current, _previous) -> None:
        """Fire on_select_fragment callback when user clicks a fragment."""
        if current is None:
            return
        fragment_id = current.data(Qt.UserRole)
        if fragment_id is not None:
            cb = getattr(self, "_on_select_fragment_cb", None)
            if cb is not None:
                try:
                    cb(fragment_id)
                except Exception:
                    pass

    def _on_seed_row_changed(self, row: int) -> None:
        """Switch painting to the marker on *row*.

        Kept as an explicit, callable seam -- nothing connects a SIGNAL
        to it any more.  `_on_marker_row_clicked` is the
        wired entry point.
        """
        if row < 0:
            return
        item = self.split_marker_list.item(row)
        if item is None:
            return
        label_id = item.data(Qt.UserRole)
        if label_id is not None:
            cb = getattr(self, "_on_select_seed_cb", None)
            if cb is not None:
                try:
                    cb(label_id)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Guided Split
    #
    # Everything below is PURE VIEW LOGIC plus one serialisable params
    # getter.  No engine call, no napari, no volume access -- the
    # orchestrator owns all of that.  Split had no `get_split_params()`
    # before this brick; it does now, and the
    # orchestrator reads the dict rather than reaching into the widgets.
    # ------------------------------------------------------------------ #

    def is_guided_split(self) -> bool:
        """True when the split needs the shared engine rather than today's code.

        This is THE gate.  While it returns False the orchestrator runs the
        legacy split path untouched, and a test asserts exactly
        that for BOTH of today's seed sources.

        **Both "(today)" options are legacy.**  `painted` and `detected` are
        the two seed sources this tool has always had, and with `shape only`
        walls they keep their existing code paths byte for byte -- `detected`
        keeps running `_detect_seeds` (EDT peaks + the histogram threshold),
        not the engine's geometry detector.  (AUDIT 2026-09-09: routing it
        into the engine made the label a lie and left only one of the two
        legacy options under the regression lock.)

        The engine is reached only by asking for something today's code
        cannot do: a trusted mask as the seeds, or a membrane signal as the
        walls.
        """
        try:
            seeds = int(self.split_seeds_from_combo.currentIndex())
            walls = int(self.split_walls_from_combo.currentIndex())
        except Exception:
            return False
        # Brick 2a: index 4 (an image detector) and EVERY wall kind past
        # `shape only` are engine-only too.  `get_split_params()["guided"]`
        # keeps its own, DIVERGENT definition -- it is in the frozen dict
        # and pinned by the golden test.  Routing is decided HERE and
        # nowhere else; do not "fix" the other one.
        return (
            seeds in IMAGE_DETECTOR_SOURCES
            or walls != WALLS_FROM_SHAPE
        )

    def guided_needs_guide_mask(self) -> bool:
        """True when the chosen seed source requires a trusted mask loaded."""
        try:
            return int(
                self.split_seeds_from_combo.currentIndex()
            ) in GUIDED_SEED_SOURCES
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # The S3 disclosure and the S5 swap
    # ------------------------------------------------------------------ #

    def _on_split_sources_toggled(self, _checked: bool) -> None:
        """The SOURCES chevron moved."""
        self._apply_split_sources_visibility()
        self._refresh_split_guided_summary()

    def _on_split_advanced_toggled(self, checked: bool) -> None:
        """The ADVANCED chevron moved (finding 10: it had no affordance)."""
        if hasattr(self, "_split_advanced_container"):
            self._split_advanced_container.setVisible(bool(checked))
        if hasattr(self, "split_advanced_btn"):
            self.split_advanced_btn.setText(
                "{} Advanced".format(
                    _CHEVRON_OPEN if checked else _CHEVRON_CLOSED)
            )

    def _apply_split_sources_visibility(self) -> None:
        """SOURCES is open when the chevron is down AND nothing owns the panel."""
        btn = getattr(self, "split_guided_summary_btn", None)
        box = getattr(self, "_split_guided_container", None)
        if btn is None or box is None:
            return
        locked = bool(getattr(self, "_split_sources_locked", False))
        box.setVisible(bool(btn.isChecked()) and not locked)
        # The three sibling sub-sections inside it.
        for sub_attr, box_attr in self._SPLIT_SUBSECTIONS:
            sub_btn = getattr(self, sub_attr, None)
            sub_box = getattr(self, box_attr, None)
            if sub_btn is None or sub_box is None:
                continue
            sub_box.setVisible(bool(sub_btn.isChecked()))
            self._refresh_chevron_text(sub_btn)

    def _set_split_sources_locked(self, locked: bool) -> None:
        """Collapse SOURCES while a solve or a review owns the panel.

        By DISABLING the summary button, never ``setChecked(False)``: the
        container is driven by that button's ``toggled`` signal, so
        unchecking it would destroy the user's disclosure state and it
        would not come back when the solve finished.
        """
        self._split_sources_locked = bool(locked)
        btn = getattr(self, "split_guided_summary_btn", None)
        if btn is not None:
            btn.setEnabled(not self._split_sources_locked)
        self._apply_split_sources_visibility()

    # ------------------------------------------------------------------ #
    # S3's three sibling sub-sections, the helper lines and the markers
    #
    # ------------------------------------------------------------------ #

    #: (chevron attribute, container attribute), in screen order.  Only
    #: one is open at a time.
    _SPLIT_SUBSECTIONS = (
        ("split_markers_btn", "_split_markers_container"),
        ("split_walls_btn", "_split_walls_group_container"),
        ("split_solve_btn", "_split_solve_container"),
    )

    #: Every control that changes WHICH markers are found.  The detectors
    #: number the markers by enumeration order over the crop, so any one
    #: of these renumbers them -- and the ticks would then select the
    #: wrong markers.  Changing one clears the list.
    _MARKER_KNOBS = (
        "split_seeds_from_combo",
        "split_seed_channel_combo",
        "split_seed_detector_combo",
        "split_seed_prep_clahe_cb",
        "split_seed_prep_zcorrect_cb",
        "split_seed_prep_bgsub_cb",
        "split_seed_prep_lift_cb",
        "split_seed_prep_denoise_cb",
        "split_seed_h_sb",
        "split_min_distance_sb",
    )

    #: Every control that changes HOW the cut is made rather than WHICH
    #: markers it starts from: the wall kind, the wall channel, the wall
    #: prepare ticks, the solver and everything under "How to cut".
    #:
    #: Moving one of these LEAVES REVIEW -- a finished result on screen
    #: was made with the old setting, and Accept beside it would commit a
    #: cut nobody previewed.  It must NOT clear the marker list: the
    #: markers are still the markers.
    _CUT_KNOBS = (
        "split_walls_from_combo",
        "split_wall_channel_combo",
        "split_wall_prep_same_cb",
        "split_wall_prep_zcorrect_cb",
        "split_wall_prep_bgsub_cb",
        "split_wall_prep_lift_cb",
        "split_wall_prep_denoise_cb",
        "split_solver_combo",
        "split_refine_guided_cb",
        "split_arbiter_cb",
        "split_arbiter_pct_sb",
        "split_target_count_cb",
        "split_target_count_sb",
        "split_refine_merge_small_cb",
        "split_min_size_sb",
        "split_wall_clip_sb",
        "split_wall_ridge_cb",
        "split_wall_smooth_sb",
        "split_rw_beta_sb",
        "split_compactness_sb",
        "split_refine_radius_sb",
        "split_refine_eps_sb",
    )

    #: What the count line says once the markers went stale.
    _MARKERS_STALE_TEXT = (
        "Markers changed - Find again, or Preview to find them fresh."
    )

    def _subsection_chevron(self, title: str, tooltip: str) -> QPushButton:
        """A checkable disclosure button that LOOKS like one.

        The title is stored as a property so the glyph can be re-rendered
        without the caller having to remember the words.
        """
        btn = QPushButton(self)
        btn.setCheckable(True)
        btn.setChecked(False)
        # The mockup's `.chev`: left-aligned, bold, quiet -- a row you
        # open, not a button you press.  A centred label on a full-width
        # button reads as an action (finding 10's other half).
        btn.setStyleSheet(PANEL_CHEVRON_STYLE)
        btn.setToolTip(tooltip)
        btn.setProperty("_epicycle_chevron_title", str(title))
        btn.setText("{} {}".format(_CHEVRON_CLOSED, title))
        return btn

    def _set_chevron_title(self, button, title: str) -> None:
        """Re-word a chevron without losing its open/closed glyph."""
        try:
            button.setProperty("_epicycle_chevron_title", str(title))
            self._refresh_chevron_text(button)
        except Exception:
            pass

    def _refresh_chevron_text(self, button) -> None:
        try:
            title = str(button.property("_epicycle_chevron_title") or "")
            if not title:
                return
            glyph = (_CHEVRON_OPEN if button.isChecked()
                     else _CHEVRON_CLOSED)
            text = "{} {}".format(glyph, title)
            # A chip that names a five-step chain is a summary like the
            # S3 line, and gets the same treatment.
            if button.property("_epicycle_elide_title"):
                self._set_elided_button_text(button, text)
            else:
                button.setText(text)
        except Exception:
            pass

    def _set_elided_button_text(self, button, full_text: str,
                                *, full_tooltip=None) -> None:
        """Show as much of *full_text* as fits; the rest is in the tooltip.

        ``full_tooltip`` is for a line whose on-screen form is already
        an abbreviation: the tooltip then carries the REAL names
        rather than a longer version of the same shorthand.
        """
        try:
            # An elided button must NOT demand the width of its own text:
            # the text is elided to that width, so the two would chase
            # each other and the line would never shrink.
            button.setSizePolicy(
                QSizePolicy.Ignored, button.sizePolicy().verticalPolicy())
            button.setProperty(_ELIDE_FULL_PROP, str(full_text))
            hint = str(button.property(_ELIDE_HINT_PROP) or "")
            shown = str(full_tooltip if full_tooltip is not None
                        else full_text)
            button.setToolTip(
                "{}\n\n{}".format(shown, hint) if hint else shown
            )
            button.installEventFilter(self._elide_guard)
            _apply_elided_text(button)
        except Exception:
            pass

    def _style_panel_list(self, lst) -> None:
        """The mockup's `.piece` rows: swatch, name, value.

        One tinted row per item with a 12 px swatch and the trailing
        value in mono grey (:class:`PanelRowDelegate`).  The item TEXT is
        untouched -- the delegate only decides which part of it is drawn
        in which font, so every test that reads a row still reads the
        same string.
        """
        from qtpy.QtCore import QSize

        try:
            lst.setStyleSheet(PANEL_LIST_STYLE)
            lst.setIconSize(QSize(12, 12))
            lst.setSpacing(0)
            lst.setUniformItemSizes(True)
            if getattr(self, "_panel_row_delegate", None) is None:
                self._panel_row_delegate = PanelRowDelegate(
                    self, separator="  {}  ".format(_DOT_SEP),
                )
            lst.setItemDelegate(self._panel_row_delegate)
        except Exception:
            pass

    @staticmethod
    def _list_row_height(lst) -> int:
        """One row of *lst*, or a sane guess before it has any."""
        try:
            row = int(lst.sizeHintForRow(0))
        except Exception:
            row = 0
        return row if row > 0 else 20

    @staticmethod
    def _list_frame(lst) -> int:
        try:
            return 2 * int(lst.frameWidth()) + 4
        except Exception:
            return 4

    def _list_height_bounds(self, lst):
        """(min, max) px the grip may take this list to."""
        row = self._list_row_height(lst)
        frame = self._list_frame(lst)
        return (_LIST_GRIP_MIN_ROWS * row + frame,
                _LIST_GRIP_MAX_ROWS * row + frame)

    def _fit_panel_list(self, lst, cap: int, prop: str = "") -> None:
        """Let a ticklist hug its rows, up to *cap* px.

        The mockup's `.list` is as tall as the pieces in it; a fixed
        reserve leaves a 70 px hole under a three-piece result and says
        nothing in it.  Purely geometric -- the cap is still the cap.

        Unless the user has dragged the grip under it, in which case
        THAT is the height, every time the list is refilled, until they
        double-click the grip (spec: owner request 2026-09-18).
        """
        try:
            chosen = self.property(prop) if prop else None
            if chosen:
                self._set_list_height_px(lst, int(chosen))
                return
            lst.setMinimumHeight(0)
            count = int(lst.count())
            if count <= 0:
                lst.setMaximumHeight(cap)
                return
            row = self._list_row_height(lst)
            frame = self._list_frame(lst)
            lst.setMaximumHeight(min(int(cap), count * row + frame))
        except Exception:
            pass

    def _set_list_height_px(self, lst, height: int) -> int:
        """Pin *lst* to *height*, clamped to the grip's bounds."""
        low, high = self._list_height_bounds(lst)
        height = max(low, min(high, int(height)))
        lst.setMinimumHeight(height)
        lst.setMaximumHeight(height)
        return height

    def _set_list_height(self, list_attr: str, prop: str,
                         height: int) -> None:
        """The grip moved: remember the height and apply it."""
        lst = getattr(self, list_attr, None)
        if lst is None:
            return
        try:
            self.setProperty(prop, self._set_list_height_px(lst, height))
        except Exception:
            pass

    def _reset_list_height(self, list_attr: str, prop: str,
                           cap: int) -> None:
        """Forget the dragged height and hug the rows again."""
        lst = getattr(self, list_attr, None)
        if lst is None:
            return
        try:
            self.setProperty(prop, None)
            lst.setMinimumHeight(0)
            self._fit_panel_list(lst, cap, prop)
        except Exception:
            pass

    def _list_with_grip(self, lst, list_attr: str, prop: str, cap: int):
        """*lst* plus its drag handle, as one thing to put in a layout."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        # Zero, deliberately: the handle is part of the list's own block
        # and is paid for out of the list's cap, not out of the page.
        lay.setSpacing(0)
        lay.addWidget(lst)
        grip = _ListHeightGrip(self, list_attr, prop, cap, parent=box)
        lay.addWidget(grip)
        setattr(self, list_attr + "_grip", grip)
        return box

    def _helper_line(self, attr: str) -> QLabel:
        """The grey one-liner under the control named *attr*.
        """
        label = QLabel(HELP_LINES.get(attr, ""), self)
        label.setWordWrap(True)
        label.setStyleSheet(_HELPER_LINE_STYLE)
        return label

    def _item_help_label(self, combo_attr: str) -> QLabel:
        """The grey line under a combo whose text swaps with the item.
        """
        label = QLabel("", self)
        label.setWordWrap(True)
        label.setStyleSheet(_HELPER_LINE_STYLE)
        self._item_help_labels[combo_attr] = label
        self._refresh_item_help(combo_attr)
        return label

    @staticmethod
    def _item_tooltip(combo_attr: str, idx: int) -> str:
        texts = ITEM_HELP.get(combo_attr, ())
        return texts[idx] if 0 <= int(idx) < len(texts) else ""

    def _apply_item_tooltips(self, combo_attr: str) -> None:
        """Put ``ITEM_HELP[combo_attr]`` on the combo's ITEMS as tooltips.

        The Refine tab's alternative to a swapping line (REFINE B3): the
        sentence is read while the list is OPEN, which is when the user is
        choosing, and it costs the page no height -- Settle and the contour
        are at their ceilings.  A greyed item's reason replaces its tooltip
        while it is greyed (Qt shows none on a disabled item anyway, so the
        reason is also on screen in the item's own text or under the combo).
        """
        combo = getattr(self, combo_attr, None)
        if combo is None:
            return
        from qtpy.QtCore import Qt as _Qt

        for idx in range(combo.count()):
            combo.setItemData(idx, self._item_tooltip(combo_attr, idx),
                              _Qt.ToolTipRole)

    def _refresh_item_help(self, combo_attr: str) -> None:
        label = self._item_help_labels.get(combo_attr)
        combo = getattr(self, combo_attr, None)
        if label is None or combo is None:
            return
        try:
            texts = ITEM_HELP.get(combo_attr, ())
            idx = int(combo.currentIndex())
            label.setText(texts[idx] if 0 <= idx < len(texts) else "")
        except Exception:
            pass

    def _on_split_subsection_toggled(self, attr: str, checked: bool) -> None:
        """Opening one sub-section closes the other two.

        Guarded against re-entry: unchecking the siblings fires their own
        ``toggled`` and would otherwise close the one just opened.
        """
        if getattr(self, "_split_subsection_syncing", False):
            return
        self._split_subsection_syncing = True
        try:
            if checked:
                for other, _box in self._SPLIT_SUBSECTIONS:
                    if other == attr:
                        continue
                    btn = getattr(self, other, None)
                    if btn is not None and btn.isChecked():
                        btn.setChecked(False)
        finally:
            self._split_subsection_syncing = False
        self._apply_split_sources_visibility()
        self._refresh_split_guided_summary()

    def _on_split_prep_toggled(self, attr: str, checked: bool) -> None:
        """One of the two "Prepare the channel first" chips moved."""
        box = {
            "split_seed_prep_btn": "_split_seed_prep_container",
            "split_wall_prep_btn": "_split_wall_prep_container",
        }.get(attr)
        container = getattr(self, box, None) if box else None
        if container is not None:
            container.setVisible(bool(checked))
        btn = getattr(self, attr, None)
        if btn is not None:
            self._refresh_chevron_text(btn)

    def _prep_ticks(self, prefix: str) -> dict:
        """The five tick-box states for one chain, as a plain dict."""
        out = {}
        for key in PREP_TICK_LABELS:
            cb = getattr(self, "{}_{}_cb".format(prefix, key), None)
            try:
                out[key] = bool(cb.isChecked()) if cb is not None else False
            except Exception:
                out[key] = False
        return out

    def _lift_token(self) -> str:
        """'gentle' -> ``gamma``, 'strong' -> ``log``.

        One knob for both chains: the wall chain either mirrors the
        marker chain outright ("Same as markers") or ticks the same five
        steps, and a second copy of this choice would let the two
        disagree with nothing on screen saying which won.
        """
        combo = getattr(self, "split_lift_mode_combo", None)
        try:
            return "log" if int(combo.currentIndex()) == 1 else "gamma"
        except Exception:
            return "gamma"

    def _lift_token_label(self) -> str:
        """The same choice in the user's word, for the params dict."""
        return "strong" if self._lift_token() == "log" else "gentle"

    def _marker_prep_chain(self) -> str:
        """The marker channel's preparation chain."""
        return chain_from_ticks(self._prep_ticks("split_seed_prep"),
                                lift_token=self._lift_token())

    def _wall_prep_chain(self) -> str:
        """The wall channel's chain.

        "Same as markers" is a MODE, not a step: while it is ticked the
        wall chain mirrors the marker chain.  CLAHE is dropped either way
        -- the wall stage applies its own through the ``membrane clip``
        knob, so keeping it here would double-apply it.
        """
        try:
            same = bool(self.split_wall_prep_same_cb.isChecked())
        except Exception:
            same = False
        ticks = (self._prep_ticks("split_seed_prep") if same
                 else self._prep_ticks("split_wall_prep"))
        return chain_from_ticks(ticks, drop_clahe=True,
                                lift_token=self._lift_token())

    def _chain_summary(self, chain: str) -> str:
        """A chain string said in the user's words, for a chip."""
        if not chain or chain == "none":
            return "none"
        words = []
        for token in str(chain).split("+"):
            key = "lift" if token in ("gamma", "log") else token
            label = PREP_TICK_LABELS.get(key)
            if label:
                words.append(label.lower())
        return ", ".join(words) if words else "none"

    def _refresh_split_prep_chips(self) -> None:
        """Both chips name what they would actually run."""
        btn = getattr(self, "split_seed_prep_btn", None)
        if btn is not None:
            self._set_chevron_title(
                btn, "Prepare the channel first: {}".format(
                    self._chain_summary(self._marker_prep_chain()))
            )
        wbtn = getattr(self, "split_wall_prep_btn", None)
        if wbtn is not None:
            self._set_chevron_title(
                wbtn, "Prepare the channel first: {}".format(
                    self._chain_summary(self._wall_prep_chain()))
            )
        # While the wall chain MIRRORS the marker chain, its own four
        # ticks are shown as what they are: not in charge.
        try:
            same = bool(self.split_wall_prep_same_cb.isChecked())
        except Exception:
            same = False
        marker = self._prep_ticks("split_seed_prep")
        keys = ("zcorrect", "bgsub", "lift", "denoise")
        was_same = bool(getattr(self, "_split_wall_prep_mirrored", False))
        if same and not was_same:
            # The TRUE edge BORROWS the four rows, so remember what
            # the user had set; otherwise unticking the box later
            # hands them the marker chain as though they chose it.
            self._split_wall_prep_saved = {
                k: self._prep_ticks("split_wall_prep").get(k, False)
                for k in keys
            }
        for key in keys:
            cb = getattr(self, "split_wall_prep_{}_cb".format(key), None)
            if cb is None:
                continue
            cb.setEnabled(not same)
            cb.blockSignals(True)
            if same:
                cb.setChecked(bool(marker.get(key)))
            elif was_same:
                # The FALSE edge gives them back.
                saved = getattr(self, "_split_wall_prep_saved", None) or {}
                cb.setChecked(bool(saved.get(key, False)))
            cb.blockSignals(False)
        self._split_wall_prep_mirrored = same

    def _wire_marker_knob_invalidation(self) -> None:
        """Every marker knob clears the found list when it moves.

        Called from the END of the builder, because two of the knobs
        (``split_seed_h_sb``, ``split_min_distance_sb``) are built in
        ADVANCED, below S3 -- wiring them inside the S3 block resolved
        them to None and connected nothing, silently.

        The wall prepare ticks are wired here too: they do not
        invalidate the markers, but the chip above them has to keep
        naming what would actually run (and "Same as markers" mirrors
        the marker chain onto them).
        """
        for attr in self._MARKER_KNOBS:
            w = getattr(self, attr, None)
            if w is None:
                continue
            for sig in (
                "currentIndexChanged", "toggled", "valueChanged",
            ):
                signal = getattr(w, sig, None)
                if signal is not None:
                    signal.connect(self._on_marker_knob_changed)
                    break
        for attr in (
            "split_wall_prep_same_cb", "split_wall_prep_zcorrect_cb",
            "split_wall_prep_bgsub_cb", "split_wall_prep_lift_cb",
            "split_wall_prep_denoise_cb",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.toggled.connect(self._on_wall_prep_tick_changed)
        # ...and every CUT knob leaves review.  Without
        # this a wall-kind change sat beside a finished result with
        # Accept live, and Accept would have committed a cut made with
        # the OLD setting.  It does NOT clear the marker list.
        for attr in self._CUT_KNOBS:
            w = getattr(self, attr, None)
            if w is None:
                continue
            for sig in (
                "currentIndexChanged", "toggled", "valueChanged",
            ):
                signal = getattr(w, sig, None)
                if signal is not None:
                    signal.connect(self._on_cut_knob_changed)
                    break

    def _on_wall_prep_tick_changed(self, *_args) -> None:
        """A wall prepare tick moved -- re-word the chip above it."""
        self._refresh_split_prep_chips()

    def _on_split_seed_detector_changed(self, index: int) -> None:
        """The detector moved -- swap its helper line, and follow hmin.

        "Membrane-enclosed interiors" reads the MEMBRANE image by
        design (it looks for the dim spaces the membrane encloses), so
        choosing it points the marker channel at the wall channel
        rather than leaving the user to discover the trick.  Only when
        a wall channel is actually chosen, and never the other way.
        """
        self._refresh_item_help("split_seed_detector_combo")
        try:
            token = SEED_DETECTORS[int(index)]
        except Exception:
            return
        if token != "hmin":
            return
        wall = getattr(self, "split_wall_channel_combo", None)
        seed = getattr(self, "split_seed_channel_combo", None)
        if wall is None or seed is None:
            return
        try:
            name = str(wall.currentText())
            if not name:
                return
            idx = seed.findText(name)
            if idx >= 0 and idx != seed.currentIndex():
                seed.setCurrentIndex(idx)
        except Exception:
            pass

    def _on_marker_knob_changed(self, *_args) -> None:
        """A marker knob moved: the found list is no longer about it.

        The ids are assigned by enumeration order over the crop, so any
        knob renumbers them and the ticks would select the WRONG markers.
        The stale sentence goes on the count line, never on S1 -- S1 is
        about the cell, not about a picker.
        """
        self._refresh_split_prep_chips()
        had = bool(getattr(self, "split_marker_list", None) is not None
                   and self.split_marker_list.count())
        self._clear_marker_list()
        if had:
            self._set_marker_count_text(self._MARKERS_STALE_TEXT)
            # The DRAWN markers went stale with the list; the
            # orchestrator owns the layer, so it is told.
            cb = getattr(self, "_on_markers_invalidated_cb", None)
            if cb is not None:
                try:
                    cb()
                except Exception:
                    pass
        self._refresh_split_guided_summary()

    def _on_cut_knob_changed(self, *_args) -> None:
        """A wall / solver / how-to-cut knob moved.

        The markers are untouched -- only the way they are flooded
        changed -- so the list stays exactly as it is.  What CANNOT stay
        is a finished result: it was made with the old setting, and
        Accept beside it would commit a cut nobody previewed.
        """
        self._refresh_split_prep_chips()
        self._refresh_split_guided_summary()
        if not getattr(self, "_split_review_active", False):
            return
        print("[RESHAPE-LOOP] cut knob moved - leaving review")
        cb = getattr(self, "_on_cut_knob_changed_cb", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def _on_marker_row_clicked(self, item) -> None:
        """A real click on a marker ROW (not on its tick).

        Fires ``on_select_seed`` so painting switches to that marker.
        The click that TOGGLED the tick is suppressed: `itemChanged`
        runs first and parks the item here, and a tick must touch
        nothing outside the panel.
        """
        ticked = getattr(self, "_marker_tick_item", None)
        self._marker_tick_item = None
        if item is None or item is ticked:
            return
        label_id = item.data(Qt.UserRole)
        if label_id is None:
            return
        cb = getattr(self, "_on_select_seed_cb", None)
        if cb is not None:
            try:
                cb(int(label_id))
            except Exception:
                pass

    def _clear_marker_list(self) -> None:
        """Empty the picker.  ``get_checked_marker_ids()`` then says None."""
        lst = getattr(self, "split_marker_list", None)
        if lst is not None:
            lst.blockSignals(True)
            lst.clear()
            lst.blockSignals(False)
            lst.setVisible(False)
        self._marker_tick_item = None
        for attr in (
            "split_marker_caption",
            "split_marker_all_btn", "split_marker_none_btn",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(False)
        self._set_marker_count_text("")
        # The verdicts belong to the rows that have just gone.
        self._marker_touching = {}
        self._set_seed_count_text(0)
        print("[MARKERS] list cleared")

    def _set_marker_count_text(self, text: str) -> None:
        label = getattr(self, "split_marker_count_label", None)
        if label is not None:
            label.setText(str(text))

    def _set_seed_count_text(self, n) -> None:
        """The "Markers: N" line above the painting rows (N = ticked rows)."""
        label = getattr(self, "split_seed_label", None)
        if label is not None:
            label.setText(_RT.MARKERS_COUNT.format(n=int(n)))

    def _touching_ids_in_list(self) -> set:
        """The listed rows the touching rule left unticked at the Find."""
        touching = getattr(self, "_marker_touching", None) or {}
        lst = getattr(self, "split_marker_list", None)
        if not touching or lst is None:
            return set()
        listed = set()
        for i in range(lst.count()):
            it = lst.item(i)
            mid = it.data(Qt.UserRole) if it is not None else None
            if mid is not None:
                listed.add(int(mid))
        return {int(k) for k in touching} & listed

    def _refresh_marker_count_line(self) -> None:
        """"Found 3 markers", or "3 of 5 ticked" once some are off.

        After a Find on markers taken from labels, while the rows the
        touching rule left unticked are exactly the unticked ones, the
        line says so instead -- "2 markers kept, 3 touching ones left
        unticked", or that none was kept.  The "Markers: N" line follows
        the ticks in every case.
        """
        lst = getattr(self, "split_marker_list", None)
        if lst is None:
            return
        if not lst.count():
            self._set_seed_count_text(0)
            return
        total = lst.count()
        kept = len(self.get_checked_marker_ids() or [])
        self._set_seed_count_text(kept)
        touching = self._touching_ids_in_list()
        unticked = set(self.get_unticked_marker_ids() or [])
        if touching and unticked == touching:
            if kept == 0:
                self._set_marker_count_text(_RT.none_kept_line(total))
            else:
                self._set_marker_count_text(_RT.MARKERS_KEPT_LINE.format(
                    kept=kept, s=_RT.plural(kept), n=len(touching),
                    s2=_RT.plural(len(touching))))
        elif kept == total:
            self._set_marker_count_text(
                _RT.MARKERS_FOUND_LINE.format(n=total, s=_RT.plural(total)))
        else:
            self._set_marker_count_text(
                _RT.MARKERS_TICKED_LINE.format(kept=kept, total=total))

    def _on_split_marker_check_changed(self, item=None, *_args) -> None:
        """A marker was ticked or unticked.

        Three things, and nothing else: the row is restyled (an unticked
        marker is dimmed and struck through), the count line and the
        summary are re-rendered, and the orchestrator is told.

        **The orchestrator is told on EVERY tick now.**  It
        used to be review-only, because leaving review was the only
        thing a tick had to do; since an unticked marker comes OFF the
        object the tick also has to lift its voxels or put them back,
        and that is just as true with no preview on screen.  The panel
        itself still touches nothing outside itself.
        """
        if item is not None:
            # The click that produced this tick must not ALSO fire
            # `on_select_seed` from `itemClicked`.
            self._marker_tick_item = item
            # ...and the restyle must not fire THIS handler again.
            # `setFont` and `setForeground` each emit `itemChanged`, so
            # one tick used to arrive here three times.  That was
            # invisible while the callback was review-only and the two
            # extra passes only re-rendered a count line; once the
            # orchestrator acts on every tick "one tick, one
            # callback" is a contract, not a nicety.  Blocking matches
            # what `_populate_marker_list` and `_set_all_marker_checks`
            # already do around their own restyles.
            lst = getattr(self, "split_marker_list", None)
            if lst is not None:
                lst.blockSignals(True)
            try:
                self._style_marker_row(item)
            finally:
                if lst is not None:
                    lst.blockSignals(False)
        self._refresh_marker_count_line()
        self._refresh_split_guided_summary()
        if getattr(self, "_split_review_active", False):
            print("[MARKERS] tick changed - leaving review")
        cb = getattr(self, "_on_marker_check_changed_cb", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    @staticmethod
    def _style_marker_row(item) -> None:
        """Dim + strike an unticked marker row.

        Unticking used to look exactly like deleting: the row simply
        lost its tick and nothing marked it out at all.

        Since the unticked marker is lifted off the object, the row is the ONLY place it can be read, because
        the marker itself has been lifted off the object by the time
        this runs.  Dimmed-and-struck means "parked, one click from
        coming back", not "gone".
        """
        if item is None:
            return
        try:
            from qtpy.QtGui import QBrush, QColor

            off = item.checkState() != Qt.Checked
            font = item.font()
            font.setStrikeOut(bool(off))
            item.setFont(font)
            item.setForeground(
                QBrush(QColor(0x77, 0x77, 0x77)) if off
                else QBrush(QColor(0xDD, 0xDD, 0xDD))
            )
        except Exception:
            pass

    def _set_all_marker_checks(self, checked: bool) -> None:
        """All / None, over the whole list in one go.

        The restyle has to be done HERE (fix pass M2).  Ticking a row by
        hand goes through `_on_split_marker_check_changed`, which dims
        and strikes it -- but these two chips run under `blockSignals`
        and reach that handler once, with no item, so after None the
        rows kept their bright, unstruck look.  That styling is what
        says WHICH markers have been lifted off the object; without it
        None reads as "deleted them all", and the rows are the only
        place left to read it, because the markers themselves are gone
        from the object by then.

        The single handler call at the end is also what parks or
        un-parks the whole list in ONE pass.
        """
        lst = getattr(self, "split_marker_list", None)
        if lst is None:
            return
        lst.blockSignals(True)
        for i in range(lst.count()):
            item = lst.item(i)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                self._style_marker_row(item)
        lst.blockSignals(False)
        self._on_split_marker_check_changed()

    def get_checked_marker_ids(self):
        """The TICKED marker ids, or ``None`` when nothing was found.

        ``None`` and ``[]`` are different answers: ``None`` means no Find
        has happened (Preview finds them fresh -- for markers taken from
        labels it leaves the touching ones out), and ``[]`` means every
        row is unticked (Preview refuses).
        """
        lst = getattr(self, "split_marker_list", None)
        if lst is None or not lst.count():
            return None
        out = []
        for i in range(lst.count()):
            item = lst.item(i)
            if item is not None and item.checkState() == Qt.Checked:
                mid = item.data(Qt.UserRole)
                if mid is not None:
                    out.append(int(mid))
        return out

    def get_unticked_marker_ids(self):
        """The UNTICKED marker ids, or ``None`` when nothing was found.

        The half `get_checked_marker_ids` cannot answer.  Parking
        needs "which rows exist AND are off", and *no list at all*
        has to stay distinguishable from *every row ticked*: with no row
        there is no tick that could put a lifted marker back, so the
        orchestrator restores instead of leaving voxels stranded.
        """
        lst = getattr(self, "split_marker_list", None)
        if lst is None or not lst.count():
            return None
        out = []
        for i in range(lst.count()):
            item = lst.item(i)
            if item is not None and item.checkState() != Qt.Checked:
                mid = item.data(Qt.UserRole)
                if mid is not None:
                    out.append(int(mid))
        return out

    def set_seed_channels(self, names, current: str = "") -> None:
        """Fill the MARKER channel combo -- the fifth channel combo.

        Same policy as the other four: filled with layer
        names, restored by the last name actually CHOSEN rather than the
        combo's live text, and index -1 on a miss so a renamed layer
        leaves a visible blank instead of silently selecting another.
        """
        combo = getattr(self, "split_seed_channel_combo", None)
        if combo is None:
            return
        prev = current or self._remember_channel_choice(combo)
        combo.blockSignals(True)
        combo.clear()
        for name in names or ():
            combo.addItem(str(name))
        self._restore_channel_choice(combo, prev)
        if not prev:
            # First fill with nothing remembered: stay blank (as Add does)
            # so the "pick a channel" refusal fires instead of a silent
            # guess of the first image layer.
            combo.setCurrentIndex(-1)
        combo.blockSignals(False)

    def _set_split_result_visible(self, visible: bool) -> None:
        """Show or hide S5, and re-home the viewport when it changes.

        The swap changes the content height under a retained scroll
        position, so every transition puts S1 back in view and focus back
        on Preview.
        """
        box = getattr(self, "_split_result_container", None)
        if box is None:
            return
        visible = bool(visible)
        was = bool(getattr(self, "_split_result_shown", False))
        self._split_result_shown = visible
        box.setVisible(visible)
        if visible != was:
            self._focus_panel_stage(
                "Reshape", reveal=box if visible else None
            )

    def _clear_split_result_rows(self) -> None:
        """Empty S5.  Called whenever the panel leaves review."""
        for attr in ("split_piece_list", "split_fragment_list"):
            lst = getattr(self, attr, None)
            if lst is None:
                continue
            lst.blockSignals(True)
            lst.clear()
            lst.blockSignals(False)
            lst.setVisible(False)
        for attr in (
            "split_piece_caption", "split_arbiter_msg_label",
            "split_engine_msg_label", "split_quality_label",
            "split_engine_error_label",
        ):
            lbl = getattr(self, attr, None)
            if lbl is not None:
                lbl.setText("")
                lbl.setVisible(False)
        for attr in ("split_anyway_btn", "split_refine_further_btn"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setVisible(False)

    def _apply_split_paint_rows_visibility(self) -> None:
        """The marker work rows are on screen while a session is running.

        ONE condition since Brick U1: the session.  They
        used to be keyed on the SOURCE as well, which is what made bug B
        possible in reverse -- a found marker could not be adjusted by
        hand, and "Add marker" was unavailable on exactly the sources
        whose finder most often misses one.  Add marker / Undo / Edit
        markers belong to the session; only the two FINDERS are keyed on
        the source (Detect Seeds on 0, Find markers on 1-4).

        Before Start every control in the block is DISABLED, so in idle
        it is 98 px of a 209 px viewport telling the user to do
        something they cannot do yet, and pushing Preview below the fold
        (owner, 2026-09-18).
        """
        box = getattr(self, "_split_paint_rows_container", None)
        if box is None:
            return
        box.setVisible(bool(getattr(self, "_split_is_active", False)))

    def _apply_split_marker_list_visibility(self) -> None:
        """The one marker list appears when it has something in it.

        An empty list is 80 px that says nothing; it does not wait for a
        session, because `update_split_ui_state(marker_ids=...)` can
        arrive from a Find before anything else has been pushed.
        """
        lst = getattr(self, "split_marker_list", None)
        if lst is None:
            return
        has = lst.count() > 0
        lst.setVisible(has)
        for attr in (
            "split_marker_caption",
            "split_marker_all_btn", "split_marker_none_btn",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(has)
        hint = getattr(self, "_split_paint_hint", None)
        if hint is not None:
            hint.setVisible(not has)

    def _fit_marker_list(self) -> None:
        """Hug the rows, up to the cap THIS stage allows (audit H2)."""
        lst = getattr(self, "split_marker_list", None)
        if lst is None:
            return
        cap = (_MARKER_LIST_REVIEW_MAX_H
               if getattr(self, "_split_review_active", False)
               else _MARKER_LIST_MAX_H)
        self._fit_panel_list(lst, cap, _MARKER_LIST_H_PROP)

    def _refresh_split_stage(self) -> None:
        """Re-derive everything the three stage flags drive.

        S1's dot colour, whether SOURCES is collapsed, and whether S5 is on
        screen -- all from ``busy`` / ``split_active`` / ``review_active``,
        so no caller has to remember to push three things at once.
        """
        if self._split_is_busy:
            state = "busy"
        elif self._split_review_active:
            state = "review"
        elif self._split_is_active:
            state = "ready"
        else:
            state = "idle"
        self._set_status_dot("_split_status_dot", state)
        self._apply_split_paint_rows_visibility()
        self._apply_split_marker_list_visibility()
        # A running SOLVE owns the panel -- you must not re-tick markers a
        # worker is already using -- so SOURCES still collapses for it.
        #
        # A finished PREVIEW does not.  Collapsing it there
        # is what made the flow a line instead of a loop: the only thing on
        # offer after a Preview was "Preview again" with the same markers,
        # and changing one meant discarding the whole session.
        #
        # FRAGMENT REFINE takes the same lock as a solve:
        # the pieces are in the volume and the Refine layer is gone, so a
        # marker tick would park voxels on a layer that does not exist
        # and a Find would write into it.
        _owned = bool(self._split_is_busy
                      or getattr(self, "_split_fragment_active", False))
        self._set_split_sources_locked(_owned)
        # ...and collapsing SOURCES is only half of it.  The marker
        # list, the All / None chips, Add marker, Undo, Edit markers and
        # the shape finder live in the SESSION block, OUTSIDE that
        # disclosure, and stayed fully live through a solve -- you could
        # untick a marker, or paint and Add another, while the worker
        # was mid-solve on the set you had just changed (fix pass M1).
        self._set_marker_work_enabled(not _owned)
        # The marker list's default height depends on the stage (H2), so
        # it is re-fitted here as well as when it is refilled -- entering
        # or leaving review does not touch the rows.
        self._fit_marker_list()
        self._refresh_split_result_visibility()

    #: Everything that can change the marker set while a session runs.
    _MARKER_WORK_WIDGETS = (
        "split_marker_list", "split_marker_all_btn", "split_marker_none_btn",
        "split_next_seed_btn", "split_undo_seed_btn",
        "split_refine_seeds_btn", "split_detect_seeds_btn",
    )

    def _set_marker_work_enabled(self, on: bool) -> None:
        """Hand the marker work to the worker, and take it back after.

        The container carries the block; the individual flags are saved
        and RESTORED rather than switched on, because several of them
        are legitimately off for their own reasons -- Undo below two
        markers, Edit markers before there is a preview -- and lighting
        them up when the answer arrives would be a second bug.
        """
        on = bool(on)
        # The snapshot comes FIRST.  `QWidget.isEnabled()` answers the
        # EFFECTIVE state, so reading it after the container has been
        # disabled records False for every child and the release would
        # then restore a panel that is uniformly dead.
        if not on and self._marker_work_was_enabled is None:
            self._marker_work_was_enabled = {
                a: bool(w.isEnabled())
                for a in self._MARKER_WORK_WIDGETS
                for w in (getattr(self, a, None),) if w is not None
            }
        for attr in ("_split_paint_rows_container", "_split_find_container"):
            box = getattr(self, attr, None)
            if box is not None:
                box.setEnabled(on)
        if not on:
            for a in self._MARKER_WORK_WIDGETS:
                w = getattr(self, a, None)
                if w is not None:
                    w.setEnabled(False)
            return
        saved = self._marker_work_was_enabled
        if saved is None:
            return
        for a, was in saved.items():
            w = getattr(self, a, None)
            if w is not None:
                w.setEnabled(bool(was))
        self._marker_work_was_enabled = None

    def _refresh_split_footer_summary(self) -> None:
        """The footer's one line says what its two buttons do to the object.

        Three states, in order (fix pass H3).  IDLE wins, because a
        reason left over from a finished session has nothing to explain.
        Then a REASON, if Accept is grey and something said why: the same
        sentence belongs in S1, here, and on Accept's tooltip,
        and this line was keyed on the reject button's TEXT alone -- a
        snap-back does not touch that button, so under a greyed-out
        Accept the footer went on reading "Accept keeps the ticked
        pieces", an instruction for a button that could not be pressed.
        Then the staged sentence.
        """
        label = getattr(self, "_split_footer_summary", None)
        btn = getattr(self, "split_reject_btn", None)
        if label is None or btn is None:
            return
        if btn.text() == "Cancel":
            label.setText("Nothing is staged yet.")
            return
        reason = getattr(self, "_split_footer_reason", "")
        label.setText(
            reason or
            "Accept keeps the ticked pieces. Discard restores the object."
        )

    def set_footer_reason(self, reason: str = "") -> None:
        """Say in the FOOTER why Accept is grey (fix pass H3).

        The sibling of `set_accept_reason`, and pushed with it, so the
        sentence the user reads under the buttons is the sentence on the
        button and the sentence in S1.  Cleared the moment Accept comes
        back.
        """
        self._split_footer_reason = str(reason or "")
        self._refresh_split_footer_summary()

    def _refresh_add_footer_summary(self) -> None:
        """The footer's one line: the orchestrator's sentence when
        it pushed one, else keyed on the secondary's word."""
        label = getattr(self, "_add_footer_summary", None)
        btn = getattr(self, "add_reject_btn", None)
        if label is None or btn is None:
            return
        from packages.epicycle_annotator import add_preview as AP

        override = getattr(self, "_add_footer_override", None)
        text = btn.text()
        if text == "Undo all":
            # REVIEW: Done (keeps the objects) + Undo all -- or the
            # orchestrator's line when a later Clear stands in the way
            n = self.add_pieces_list.count() if hasattr(
                self, "add_pieces_list") else 0
            label.setText(override or AP.foot_review(n))
        elif text == "Discard":
            label.setText(AP.ADD_FOOT_CANDIDATES)
        elif override:
            label.setText(override)
        elif getattr(self, "_add_ui", {}).get("box"):
            label.setText(AP.ADD_FOOT_BOX)
        else:
            label.setText(AP.ADD_FOOT_IDLE)
        # the secondary's tooltip is the line under it, in every state
        # but REVIEW (Undo all keeps its own) -- so it never names a word
        # the button does not show, and in a "+ Add another" box it says
        # Close box goes back to the review
        if text != "Undo all":
            btn.setToolTip(label.text())

    def _refresh_refine_footer_summary(self) -> None:
        """Same line, same job, on the Refine tab."""
        label = getattr(self, "_refine_footer_summary", None)
        btn = getattr(self, "smooth_reject_btn", None)
        if label is None or btn is None:
            return
        if btn.text() == "Discard":
            # the chain is the feature, so the footer says what
            # Discard actually undoes -- ALL of it, not the last step.
            label.setText(
                "Accept keeps the new shape. Discard puts the object back "
                "as it was before the first step."
            )
        else:
            # the only other place the menu is named.
            label.setText(REFINE_FOOTER_IDLE)

    def _refresh_split_result_visibility(self) -> None:
        """S5 is on screen exactly while it has something to say."""
        def _shown(attr):
            w = getattr(self, attr, None)
            return bool(w is not None and not w.isHidden())

        def _filled(attr):
            w = getattr(self, attr, None)
            return bool(w is not None and not w.isHidden() and w.count())

        self._set_split_result_visible(
            self._split_review_active
            or _filled("split_piece_list")
            or _filled("split_fragment_list")
            or _shown("split_piece_caption")
            or _shown("split_quality_label")
            or _shown("split_arbiter_msg_label")
            or _shown("split_anyway_btn")
            or _shown("split_engine_msg_label")
            or _shown("split_engine_error_label")
            or _shown("split_refine_further_btn")
        )

    def _refresh_split_guided_summary(self) -> None:
        """Re-render S3's one-line collapsed summary from the pickers.

        It carries a CHEVRON (the button used to render as an ordinary
        button whose label happened to be a summary, with nothing to say
        that clicking it opened anything), and it is ELIDED to the
        button's own width with the whole line in the tooltip: at the
        340 px dock it does not fit at almost any setting, and a clipped
        line silently drops the half that names the walls.
        """
        if not hasattr(self, "split_guided_summary_btn"):
            return
        try:
            seeds = SEEDS_FROM_LABELS[
                int(self.split_seeds_from_combo.currentIndex())
            ]
            walls = WALLS_FROM_LABELS[
                int(self.split_walls_from_combo.currentIndex())
            ]
        except Exception:
            return
        try:
            short_seeds = SEEDS_FROM_SHORT[
                int(self.split_seeds_from_combo.currentIndex())]
            short_walls = WALLS_FROM_SHORT[
                int(self.split_walls_from_combo.currentIndex())]
        except Exception:
            short_seeds, short_walls = seeds, walls
        ticked = ""
        ticked_long = ""
        lst = getattr(self, "split_marker_list", None)
        if lst is not None and lst.count():
            n_ticked = len(self.get_checked_marker_ids() or [])
            ticked = " ({})".format(n_ticked)
            ticked_long = " ({} ticked)".format(n_ticked)
        chevron = (
            _CHEVRON_OPEN if self.split_guided_summary_btn.isChecked()
            else _CHEVRON_CLOSED
        )
        # "expect 3" rides the COLLAPSED line.  It
        # is the one survivor of a session that can silently produce a
        # WRONG cut rather than an inconvenient one -- a 4-nucleus object
        # inheriting "expect 2" merges two real objects down and the
        # quality gate stays quiet, because the stage did what it was
        # told.  So while it is armed, it is on screen even collapsed.
        expect = ""
        cb = getattr(self, "split_target_count_cb", None)
        sb = getattr(self, "split_target_count_sb", None)
        if cb is not None and sb is not None and cb.isChecked():
            expect = "{}expect {}".format(_SUMMARY_SEP, int(sb.value()))
        # SHORT on the button -- there are 307 px of it -- and the
        # real names in the tooltip.  A summary nobody can read is
        # not one.
        self._set_elided_button_text(
            self.split_guided_summary_btn,
            "{} Sources - markers: {}{}{}walls: {}{}".format(
                chevron, short_seeds, ticked, _SUMMARY_SEP, short_walls,
                expect),
            full_tooltip="Markers: {}{}{}Walls: {}{}".format(
                seeds, ticked_long, _SUMMARY_SEP, walls, expect),
        )

    def _on_split_seeds_from_changed(self, index: int) -> None:
        """Reveal the guide picker for the two guided seed sources.

        The RUN choice and the FILE choice are different questions, so only
        the one that applies is shown -- index 2 and index 3 used to reveal
        the identical block (finding 9).  Detect is hidden for both, because
        a mask source ignores whatever it would find (#5).
        """
        idx = int(index)
        show_guide = idx in GUIDED_SEED_SOURCES
        show_image = idx == SEEDS_FROM_IMAGE
        if hasattr(self, "_split_guide_container"):
            self._split_guide_container.setVisible(show_guide)
        if hasattr(self, "_split_guide_run_row"):
            self._split_guide_run_row.setVisible(idx == SEEDS_FROM_RUN_LABELS)
        if hasattr(self, "_split_guide_file_row"):
            self._split_guide_file_row.setVisible(
                idx == SEEDS_FROM_EXTERNAL_MASK
            )
        if hasattr(self, "_split_image_seed_container"):
            self._split_image_seed_container.setVisible(show_image)
        self._refresh_guide_load_enabled()
        if hasattr(self, "_split_find_container"):
            # ONE FINDER PER CARD.  Find markers is the
            # finder for sources 1-4 -- index 1 joined them, because
            # "Shape thickness peaks" is a finder too and used to show
            # the painting rows and then silently discard what was
            # painted (bug B).
            self._split_find_container.setVisible(
                idx in FIND_MARKER_SOURCES)
        if hasattr(self, "split_detect_seeds_btn"):
            # ...and Detect Seeds is source 0's finder, and only source
            # 0's.  It was hidden for 2-4 already; index 1 loses it now.
            self.split_detect_seeds_btn.setVisible(
                idx in DETECT_SEED_SOURCES)
        if hasattr(self, "_split_paint_rows_container"):
            # The rows themselves are keyed on the SESSION, not on the
            # source: every source can gain a marker by hand.
            self._apply_split_paint_rows_visibility()
            if idx in IMAGE_DETECTOR_SOURCES and hasattr(
                    self, "split_edt_cb"):
                # The distance heatmap is about the shape, so an image
                # source has no use for it; and a checked box left
                # behind leaves its overlay on the volume.
                self.split_edt_cb.setChecked(False)
        self._refresh_item_help("split_seeds_from_combo")
        self._refresh_wall_item_availability()
        self._refresh_split_guided_summary()
        cb = getattr(self, "_on_seeds_from_changed_cb", None)
        if cb is not None:
            try:
                cb(int(index))
            except Exception:
                pass

    def _on_split_walls_from_changed(self, index: int) -> None:
        """Reveal the channel knobs, and with them the arbiter.

        The arbiter is only meaningful against an image-derived elevation,
        so it is hidden -- not merely disabled -- whenever the
        walls come from the shape alone.  The test is the wall KIND, not
        the single index 1 it used to be: every one of the seven kinds
        appended in Brick 2a is on one side of it or the other.
        """
        show_walls = int(index) in IMAGE_DERIVED_WALL_INDICES
        if hasattr(self, "_split_walls_container"):
            self._split_walls_container.setVisible(show_walls)
        # The two promoted rows that need image evidence live in the
        # How-to-cut card now, so they have their own container -- the
        # rule is the same one, read off the same set.
        if hasattr(self, "_split_cut_image_container"):
            self._split_cut_image_container.setVisible(show_walls)
        # "Prepare the channel first" prepares the CHANNEL, and a shape
        # wall has none -- the chip was the last thing on screen still
        # offering to clean an image nothing would read (M3).  It is
        # CLOSED as well as hidden: a chip left open behind an invisible
        # one reappears open on the next image wall, showing five ticks
        # the user never re-opened.
        for _attr in ("split_wall_prep_btn", "_split_wall_prep_container"):
            _w = getattr(self, _attr, None)
            if _w is not None:
                _w.setVisible(show_walls)
        if not show_walls:
            _chip = getattr(self, "split_wall_prep_btn", None)
            if _chip is not None and _chip.isChecked():
                # Its own toggled handler hides the container and
                # re-renders the chevron; nothing else to undo.
                _chip.setChecked(False)
        if not show_walls:
            # Never leave a checked box behind an invisible one: the
            # engine refuses an arbiter without image evidence outright,
            # and a snap onto a shape elevation is a setting with nothing
            # on screen to turn it off.
            for _attr in ("split_arbiter_cb", "split_refine_guided_cb"):
                _cb = getattr(self, _attr, None)
                if _cb is not None:
                    _cb.setChecked(False)
        self._refresh_item_help("split_walls_from_combo")
        self._refresh_split_prep_chips()
        self._refresh_split_guided_summary()
        cb = getattr(self, "_on_walls_from_changed_cb", None)
        if cb is not None:
            try:
                cb(int(index))
            except Exception:
                pass

    def _refresh_wall_item_availability(self) -> None:
        """Shape-thickness markers need a SHAPE wall.

        The engine hands the WALL elevation to the ``distance`` detector,
        so ``No wall`` finds nothing at all (measured: 0 markers on a
        10x20x20 box, against 1 for a shape wall) and ``Between the
        spines`` is quietly a different detector than its label says.
        Both are disabled while that marker source is chosen, with the
        reason on the item.
        """
        combo = getattr(self, "split_walls_from_combo", None)
        seeds = getattr(self, "split_seeds_from_combo", None)
        if combo is None or seeds is None:
            return
        try:
            shape_markers = int(
                seeds.currentIndex()) == SEEDS_FROM_DETECTED
            model = combo.model()
        except Exception:
            return
        for idx in range(combo.count()):
            item = model.item(idx) if hasattr(model, "item") else None
            if item is None:
                continue
            blocked = shape_markers and idx in SHAPE_MARKER_FORBIDDEN_WALLS
            try:
                item.setEnabled(not blocked)
                item.setToolTip(
                    "Shape thickness peaks needs a shape wall."
                    if blocked else ""
                )
            except Exception:
                pass
        if shape_markers and combo.currentIndex() in (
                SHAPE_MARKER_FORBIDDEN_WALLS):
            combo.setCurrentIndex(WALLS_FROM_SHAPE)

    def _on_split_piece_check_changed(self, *_args) -> None:
        """A piece checkbox toggled -- retitle Accept and tell the orchestrator.

        Unticking does NOT re-solve; it only narrows the commit.
        """
        k = len(self.get_checked_split_pieces())
        if hasattr(self, "split_accept_btn") and self.split_piece_list.count():
            self.split_accept_btn.setText(
                "Accept ({})".format(k) if k else "Accept"
            )
        cb = getattr(self, "_on_split_piece_check_cb", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def get_checked_split_pieces(self) -> list:
        """Piece ids currently TICKED (what Accept will make into cells)."""
        out = []
        if not hasattr(self, "split_piece_list"):
            return out
        for i in range(self.split_piece_list.count()):
            it = self.split_piece_list.item(i)
            if it is not None and it.checkState() == Qt.Checked:
                pid = it.data(Qt.UserRole)
                if pid is not None:
                    out.append(int(pid))
        return out

    def set_guide_sources(self, sources) -> None:
        """Fill the guide picker.  ``sources`` are ``reshape_guide.GuideSource``.

        The full object goes in ``Qt.UserRole`` so the orchestrator never has
        to parse the label back apart.
        """
        if not hasattr(self, "split_guide_source_combo"):
            return
        combo = self.split_guide_source_combo
        combo.blockSignals(True)
        combo.clear()
        for src in sources or ():
            combo.addItem(getattr(src, "label", str(src)), src)
        combo.blockSignals(False)
        if not combo.count():
            combo.addItem(_RT.NO_RUN_ITEM, None)
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)
        self._refresh_guide_load_enabled()

    def _refresh_guide_load_enabled(self) -> None:
        """Load mask is grey while "another run" shows an empty picker.

        There is nothing it could load then; on "a mask file" it is live
        again, because Browse is what fills it there.
        """
        btn = getattr(self, "split_guide_load_btn", None)
        seeds = getattr(self, "split_seeds_from_combo", None)
        combo = getattr(self, "split_guide_source_combo", None)
        if btn is None or seeds is None:
            return
        empty_run_picker = (
            int(seeds.currentIndex()) == SEEDS_FROM_RUN_LABELS
            and combo is not None
            and not any(combo.itemData(i) is not None
                        for i in range(combo.count()))
        )
        btn.setEnabled(not empty_run_picker)

    def selected_guide_source(self):
        """The picked ``GuideSource``, or None."""
        if not hasattr(self, "split_guide_source_combo"):
            return None
        try:
            return self.split_guide_source_combo.currentData()
        except Exception:
            return None

    def _on_split_smooth_mode_changed(self, idx) -> None:
        on = int(idx) != SMOOTH_MODE_OFF
        self.split_auto_smooth_cb.setChecked(on)
        self._smooth_sigma_container.setVisible(on)

    def get_split_smooth_mode(self) -> int:
        """0 = off, 1 = smooth the pieces, 2 = smooth and keep the object."""
        return int(self.split_smooth_mode_combo.currentIndex())

    def get_split_params(self) -> dict:
        """The whole Split recipe as a plain, serialisable dict.

        Keys map one-to-one onto ``ReshapeSpec`` fields and params, so the
        orchestrator builds a spec without reaching into any widget.  A knob
        that reaches nothing must not exist, so every
        key here is consumed.
        """
        def _idx(name, default=0):
            w = getattr(self, name, None)
            try:
                return int(w.currentIndex())
            except Exception:
                return default

        def _val(name, default):
            w = getattr(self, name, None)
            try:
                return type(default)(w.value())
            except Exception:
                return default

        def _chk(name, default=False):
            w = getattr(self, name, None)
            try:
                return bool(w.isChecked())
            except Exception:
                return default

        def _txt(name, default=""):
            w = getattr(self, name, None)
            try:
                return str(w.currentText())
            except Exception:
                return default

        seeds_from = _idx("split_seeds_from_combo", SEEDS_FROM_PAINTED)
        walls_from = _idx("split_walls_from_combo", WALLS_FROM_SHAPE)
        solver_idx = _idx("split_solver_combo", 0)
        # The tick is the "off" switch now (it used to be a 0 in the spin
        # box, which read as "zero cells").
        target_count = (
            _val("split_target_count_sb", 2)
            if _chk("split_target_count_cb", False) else 0
        )
        # The minimum IS "automatic" -- the engine wants None for that.
        seed_thresh = _val("split_seed_thresh_sb", 0.0)
        return {
            "seeds_from": seeds_from,
            "walls_from": walls_from,
            # --- Brick 2a: the five ADDITIONS.  Every pre-existing key
            # below keeps its value for every pre-existing combination
            # (pinned by test_split_params_golden).
            "seed_detector": _idx("split_seed_detector_combo", 0),
            "seed_channel": _txt("split_seed_channel_combo"),
            "seed_preproc": self._marker_prep_chain(),
            "wall_preproc": self._wall_prep_chain(),
            "search_padding": _val("split_search_padding_sb", 15),
            "pieces_cap": _val("split_pieces_cap_sb", 10),
            # --- Brick 2b: the eleven remaining engine knobs.  Every
            # default here is `reshape.spec.DEFAULT_PARAMS`'s own, so a
            # user who never opens Advanced gets the same answer as
            # before (pinned in test_reshape_split_widget).
            "seed_smooth": _val("split_seed_smooth_sb", 2.0),
            "max_sigma_um": _val("split_max_sigma_sb", 6.0),
            "seed_thresh": seed_thresh if seed_thresh > 0 else None,
            "seed_clip": _val("split_seed_clip_sb", 0.02),
            "lift_mode": self._lift_token_label(),
            "seed_gamma": _val("split_seed_gamma_sb", 0.5),
            "seed_bg": _val("split_seed_bg_sb", 15),
            "seed_denoise": _val("split_seed_denoise_sb", 2),
            "smooth": _val("split_wall_smooth_sb", 1.0),
            "refine_radius": _val("split_refine_radius_sb", 4),
            "refine_eps": _val("split_refine_eps_sb", 0.01),
            "guided": (
                seeds_from != SEEDS_FROM_PAINTED or walls_from != WALLS_FROM_SHAPE
            ),
            "wall_channel": _txt("split_wall_channel_combo"),
            "clip": _val("split_wall_clip_sb", 0.01),
            "ridge": _chk("split_wall_ridge_cb", True),
            "arbiter": _chk("split_arbiter_cb", False),
            "arbiter_percentile": _val("split_arbiter_pct_sb", 75.0),
            "solver": SPLIT_SOLVERS[
                solver_idx if 0 <= solver_idx < len(SPLIT_SOLVERS) else 0
            ],
            "rw_beta": _val("split_rw_beta_sb", 130.0),
            "compactness": _val("split_compactness_sb", 0.0),
            "h": _val("split_seed_h_sb", 0.15),
            "min_distance_um": _val("split_min_distance_sb", 5.0),
            "refine_guided": _chk("split_refine_guided_cb", False),
            "refine_merge_small": _chk("split_refine_merge_small_cb", False),
            "min_size": _val("split_min_size_sb", 20),
            # An unticked "I expect exactly n cells" means "do not force
            # a count" -- the engine wants None for that, never 0.
            "target_count": target_count if target_count > 0 else None,
        }

    #: Where a channel combo remembers the last layer name the user (or the
    #: orchestrator) actually chose.  A Qt property rather than an attribute
    #: so it travels with the widget and cannot be missed by a refresher.
    _CHANNEL_CHOICE_PROP = "_epicycle_channel_choice"

    @classmethod
    def _remember_channel_choice(cls, combo) -> str:
        """The name a refill must restore: the live selection when there is
        one, otherwise the last real choice.

        Reading `currentText()` alone is not enough, and that was the live
        bug behind the driver's d6/d7/d8: the FIRST refresh after a rename
        blanks the combo (correctly), and the SECOND one then reads "" ,
        takes it for "nothing was ever chosen", and lets Qt's addItems
        auto-selection at index 0 stand -- so the empty state healed into a
        silently WRONG channel, and every downstream "that layer is gone"
        error became unreachable because the combo named a layer that did
        exist.  Remembering the choice across the blank state is what keeps
        the refusal reachable (and is also what lets a rename-BACK restore
        the selection).
        """
        try:
            if combo.currentIndex() >= 0 and combo.currentText():
                combo.setProperty(cls._CHANNEL_CHOICE_PROP, combo.currentText())
            return str(combo.property(cls._CHANNEL_CHOICE_PROP) or "")
        except Exception:
            return ""

    @classmethod
    def _restore_channel_choice(cls, combo, prev: str) -> None:
        """Re-select *prev* after a refill, or select NOTHING.

        All four channel combos do clear() -> addItems() -> findText(prev),
        and on a miss they used to do nothing -- but Qt has already
        auto-selected index 0 inside the blockSignals window, so a renamed
        or removed layer silently reselected THE FIRST IMAGE LAYER and the
        tool segmented against the wrong channel with no message.  An empty index is the visible empty state the explicit
        errors can then report.

        A first fill -- nothing ever chosen, nothing remembered -- has
        nothing to restore, so index 0 stands there for Add's main combo
        only; the Refine, wall and marker setters blank it afterwards.
        """
        if not prev:
            return
        try:
            combo.setProperty(cls._CHANNEL_CHOICE_PROP, prev)
        except Exception:
            pass
        combo.setCurrentIndex(combo.findText(prev))

    def set_wall_channels(self, names, current: str = "") -> None:
        """Fill the membrane-channel combo from the viewer's image layers."""
        combo = getattr(self, "split_wall_channel_combo", None)
        if combo is None:
            return
        # An empty `current` falls back to the REMEMBERED choice, so passing
        # "" (as the orchestrator does at Start) preserves the selection --
        # and, after a rename, keeps the combo blank instead of healing it
        # into the first layer.
        prev = current or self._remember_channel_choice(combo)
        combo.blockSignals(True)
        combo.clear()
        for name in names or ():
            combo.addItem(str(name))
        self._restore_channel_choice(combo, prev)
        if not prev:
            # First fill with nothing remembered: stay blank (as Add does)
            # so the "pick a channel" refusal fires instead of a silent
            # guess of the first image layer.
            combo.setCurrentIndex(-1)
        combo.blockSignals(False)

    def update_split_guided_ui(
        self,
        *,
        pieces=None,
        piece_sizes=None,
        piece_flags=None,
        caption=None,
        arbiter_message=None,
        split_anyway_visible=None,
        engine_messages=None,
        engine_error=None,
        quality_notes=None,
        guide_note=None,
        guide_loaded=None,
        busy=None,
        busy_text=None,
        busy_progress=None,
        blocked_reason=None,
        guide_note_kind=None,
    ) -> None:
        """Partial, None-guarded update of every guided-Split affordance.

        ``guide_note_kind`` is how ``guide_note`` is drawn: ``"info"`` in
        grey, ``"warn"`` in amber (a refusal or a caution).  Not passed,
        the note keeps the amber it always had.

        Mirrors ``update_add_ui_state``'s contract: every parameter defaults
        to None and only the ones passed are applied, so a caller can nudge
        one thing cheaply.
        """
        from qtpy.QtGui import QColor

        # The quality notes are read BEFORE the pieces: rule (c) names the
        # slivers, and the piece list unticks exactly what it is told.
        # The 0.2 x median arithmetic used to live here, in three copies;
        # it lives in `reshape.quality` now -- ONE rule.
        if quality_notes is not None:
            self._apply_quality_notes(quality_notes)

        if pieces is not None and hasattr(self, "split_piece_list"):
            sizes = {int(k): int(v) for k, v in (piece_sizes or {}).items()}
            flags = {int(k): str(v) for k, v in (piece_flags or {}).items()}
            frag_ids = set(getattr(self, "_split_quality_untick", ()) or ())
            self.split_piece_list.blockSignals(True)
            self.split_piece_list.clear()
            for i, pid in enumerate(pieces):
                pid = int(pid)
                sz = sizes.get(pid)
                flag = flags.get(pid, "")
                is_frag = pid in frag_ids
                text = "Piece {}".format(i + 1)
                tip = "Piece {} (id {})".format(i + 1, pid)
                if sz is not None:
                    text += "  {}  {:,} vx".format(_DOT_SEP, sz)
                    tip += " -- {:,} voxels".format(sz)
                if flag:
                    text += "  [{}]".format(flag)
                    tip += " -- flagged {}".format(flag)
                if is_frag:
                    text += "  (small)"
                    tip += " -- small, likely a sliver (unticked)"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, pid)
                try:
                    item.setIcon(self._seed_swatch_icon(pid))
                except Exception:
                    pass
                item.setToolTip(tip)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked if is_frag else Qt.Checked)
                if is_frag or flag:
                    item.setForeground(QColor("#e0a030"))
                self.split_piece_list.addItem(item)
            self.split_piece_list.blockSignals(False)
            self._fit_panel_list(
                self.split_piece_list, _PIECE_LIST_MAX_H,
                _PIECE_LIST_H_PROP)
            self.split_piece_list.setVisible(len(pieces) > 0)
            # itemChanged was blocked during population -- set the initial
            # Accept (K) count by hand, exactly as the Add list does.
            self._on_split_piece_check_changed()

        if caption is not None and hasattr(self, "split_piece_caption"):
            self.split_piece_caption.setText(str(caption))
            self.split_piece_caption.setVisible(bool(caption))

        if engine_error is not None and hasattr(
            self, "split_engine_error_label"
        ):
            text = str(engine_error)
            self.split_engine_error_label.setText(text)
            self.split_engine_error_label.setVisible(bool(text))

        if quality_notes is not None and hasattr(self, "split_quality_label"):
            text = "\n".join(
                self._quality_note_text(n) for n in quality_notes)
            self.split_quality_label.setText(text)
            self.split_quality_label.setVisible(bool(text))

        if arbiter_message is not None and hasattr(
            self, "split_arbiter_msg_label"
        ):
            self.split_arbiter_msg_label.setText(str(arbiter_message))
            self.split_arbiter_msg_label.setVisible(bool(arbiter_message))

        if split_anyway_visible is not None and hasattr(
            self, "split_anyway_btn"
        ):
            self.split_anyway_btn.setVisible(bool(split_anyway_visible))

        if engine_messages is not None and hasattr(
            self, "split_engine_msg_label"
        ):
            text = "\n".join(str(m) for m in engine_messages)
            self.split_engine_msg_label.setText(text)
            self.split_engine_msg_label.setVisible(bool(text))

        if guide_note is not None and hasattr(self, "split_guide_note_label"):
            self.split_guide_note_label.setStyleSheet(
                PANEL_HELP_STYLE if guide_note_kind == _RT.GUIDE_NOTE_INFO
                else PANEL_WARN_STYLE)
            self.split_guide_note_label.setText(str(guide_note))
            self.split_guide_note_label.setVisible(bool(guide_note))

        if guide_loaded is not None and hasattr(self, "split_guide_show_cb"):
            self.split_guide_show_cb.setEnabled(bool(guide_loaded))

        if busy is not None and hasattr(self, "_split_busy_container"):
            self._split_busy_container.setVisible(bool(busy))
            # "Stop waiting" appears BESIDE Preview only while there is
            # something to stop.
            if hasattr(self, "split_cancel_preview_btn"):
                self.split_cancel_preview_btn.setVisible(bool(busy))
            # SOLVING collapses SOURCES -- by disabling the chevron, never
            # by unchecking it.
            self._split_is_busy = bool(busy)
            self._refresh_split_stage()
            if not busy and hasattr(self, "split_busy_bar"):
                self.split_busy_bar.setRange(0, 0)

        if busy_progress is not None and hasattr(self, "split_busy_bar"):
            done, total = busy_progress
            if total and total > 0:
                self.split_busy_bar.setRange(0, int(total))
                self.split_busy_bar.setValue(int(done))
            else:
                self.split_busy_bar.setRange(0, 0)

        if busy_text is not None and hasattr(self, "split_status_label"):
            self.split_status_label.setText(str(busy_text))

        if blocked_reason is not None and hasattr(self, "split_accept_btn"):
            # "Accept blocked by a lock, naming which" -- a silently dead
            # button is worse than a disabled one that says why.  It used to
            # ALSO rewrite the Start button's tooltip with text about seed
            # painting, which the S2 button no longer says; the reason goes
            # to the STATUS line instead, which is what S1 is for.
            reason = str(blocked_reason)
            self.split_accept_btn.setToolTip(
                reason or (
                    "Write the ticked pieces into the volume as separate "
                    "objects. Anything you unticked stays part of the "
                    "original object -- it is never deleted."
                )
            )
            if reason and hasattr(self, "split_status_label"):
                self.split_status_label.setText(reason)
                self._set_status_dot("_split_status_dot", "busy")

        # S5 is a container now, not five loose rows below the buttons that
        # act on them (finding 2).  Recompute its visibility from whatever
        # this call left on screen.
        if any(
            v is not None for v in (
                pieces, caption, arbiter_message, split_anyway_visible,
                engine_messages, engine_error, quality_notes,
            )
        ):
            self._refresh_split_result_visibility()

    @staticmethod
    def _quality_note_text(note) -> str:
        """One note's sentence, whether it arrived as an object or data."""
        if isinstance(note, dict):
            return str(note.get("text", ""))
        return str(getattr(note, "text", note))

    @staticmethod
    def _quality_note_kind(note) -> str:
        if isinstance(note, dict):
            return str(note.get("kind", ""))
        return str(getattr(note, "kind", ""))

    @staticmethod
    def _quality_note_ids(note):
        if isinstance(note, dict):
            return tuple(note.get("piece_ids", ()) or ())
        return tuple(getattr(note, "piece_ids", ()) or ())

    def _apply_quality_notes(self, notes) -> None:
        """Remember which pieces the gate unticked for the user.

        A sliver is the ONE note kind that names pieces to act on; the
        others are read, not acted on.  Kept as a set on the widget so
        the piece list, which is filled in the same call, can consult it
        without the caller having to pass the same information twice.
        """
        # Only a SLIVER note names pieces to act on.  The other three
        # kinds are read, not acted on -- there is no piece to untick for
        # a coverage shortfall, and unticking one for a lost marker or a
        # short count would be the tool deciding something it has just
        # said it cannot decide.
        untick = set()
        for note in notes or ():
            if self._quality_note_kind(note) == "sliver":
                untick.update(int(v) for v in self._quality_note_ids(note))
        self._split_quality_untick = untick

    def clear_split_guided_ui(self) -> None:
        """Return every guided affordance to its idle, empty state.

        The MARKER list goes with it, so ``get_checked_marker_ids()``
        answers None again: a Discard on cell 139 must not leave ids
        behind for cell 140 to be cut with.
        """
        self.update_split_guided_ui(
            pieces=[], piece_sizes={}, piece_flags={}, caption="",
            arbiter_message="", split_anyway_visible=False,
            engine_messages=[], engine_error="", quality_notes=[],
            busy=False,
        )
        self._clear_marker_list()
        self._set_marker_busy(False)
        self.set_accept_reason("")
        self.set_footer_reason("")
        # Idle is not a review: S5 goes away and Preview stops offering to
        # do it "again" (#11/#12).
        #
        # ...and it is not a FRAGMENT REFINE either (audit H1, U4).  This
        # method IS the panel call a patch switch makes
        # (`_teardown_reshape_for_switch`), and it pushed `review_active`
        # alone -- so a switch out of a fragment session left
        # `_split_fragment_active` True and S3 plus the whole marker
        # block dead for the REST of the annotator session, on the next
        # object and every object after it.  The session is over; every
        # flag that describes it goes down together.
        self.update_split_ui_state(
            review_active=False, fragment_refine_active=False)
        if hasattr(self, "split_accept_btn"):
            self.split_accept_btn.setText("Accept")

    # ------------------------------------------------------------------ #
    # Brick U3 -- "Show what the tool sees"
    # ------------------------------------------------------------------ #
    def walls_legend_text(self) -> str:
        """The sentence under the tick, worded from the WALL KIND.

        Read off the live combo rather than off the result, and the two
        cannot drift: any change in "How to cut" or in the wall kind is
        a ``_CUT_KNOBS`` change, which leaves review, which takes the
        picture and the tick with it.

        Keyed on the wall kind's NAME, not on a bare index, so appending
        a kind cannot silently give it the wrong physics -- it would get
        the pinch sentence, which is at least the default the picker
        starts on.
        """
        try:
            idx = int(self.split_walls_from_combo.currentIndex())
        except Exception:
            idx = WALLS_FROM_SHAPE
        if idx in IMAGE_DERIVED_WALL_INDICES:
            return WALLS_LEGEND_IMAGE
        kind = WALL_KINDS[idx] if 0 <= idx < len(WALL_KINDS) else "distance"
        if kind == "skeleton":
            return WALLS_LEGEND_SPINES
        if kind == "none":
            return WALLS_LEGEND_FLAT
        return WALLS_LEGEND_PINCH

    def is_show_walls_checked(self) -> bool:
        """Is the wall picture wanted right now?"""
        cb = getattr(self, "split_show_walls_cb", None)
        return bool(cb is not None and cb.isChecked())

    def set_walls_toggle_enabled(self, enabled: bool,
                                 reason: str = "") -> None:
        """Offer the tick, or say why it is grey -- ON SCREEN.

        **Qt shows no tooltip on a disabled widget** (fix pass H5).  Every
        reason this control had for being grey was therefore written
        somewhere nobody could read it: hover a disabled checkbox and
        nothing happens, so "why is that tick grey" had no answer at all.

        So an explicit reason is ALSO written into the legend's slot --
        the one line under the tick, which is on screen exactly when S5
        is (i.e. in review, which is the only state where a grey tick is
        a question worth asking).  Outside review the card is hidden and
        the line costs nothing.

        A reason of ``""`` means "there is nothing worth saying" and
        leaves the slot empty; the tooltip still carries the default for
        the enabled-again case.  Enabling clears the reason and hands the
        slot back to the legend, which belongs to the box being TICKED.
        """
        cb = getattr(self, "split_show_walls_cb", None)
        if cb is None:
            return
        cb.setEnabled(bool(enabled))
        cb.setToolTip(
            WALLS_TOGGLE_HELP if enabled
            else (str(reason) or WALLS_TOGGLE_DISABLED)
        )
        legend = getattr(self, "split_walls_legend", None)
        if legend is None:
            return
        if enabled:
            checked = self.is_show_walls_checked()
            legend.setText(self.walls_legend_text() if checked else "")
            legend.setVisible(bool(checked))
        else:
            text = str(reason or "")
            legend.setText(text)
            legend.setVisible(bool(text))

    def clear_walls_toggle(self) -> None:
        """Untick, disable and take the legend away.  SILENT.

        ``blockSignals`` is the point: the ORCHESTRATOR removes the
        layer explicitly at each of the six exits, so a programmatic
        untick that also fired the callback would make the order between
        the two load-bearing for no gain.
        """
        cb = getattr(self, "split_show_walls_cb", None)
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
            cb.setEnabled(False)
            cb.setToolTip(WALLS_TOGGLE_DISABLED)
        legend = getattr(self, "split_walls_legend", None)
        if legend is not None:
            legend.setText("")
            legend.setVisible(False)

    def set_walls_toggle_checked(self, checked: bool) -> None:
        """Put the tick back up SILENTLY after a re-Preview.

        The orchestrator holds the intent (``_walls_wanted``), because
        pressing "Preview again" pushes ``review_active=False`` and that
        returns every result-shaped control to idle -- this tick with
        them.  When the new result lands with the picture already
        rebuilt, the box has to agree with the viewer again, and it must
        do so without asking the orchestrator to build it twice.
        """
        cb = getattr(self, "split_show_walls_cb", None)
        if cb is None:
            return
        cb.blockSignals(True)
        cb.setChecked(bool(checked))
        cb.blockSignals(False)
        legend = getattr(self, "split_walls_legend", None)
        if legend is not None:
            legend.setText(self.walls_legend_text() if checked else "")
            legend.setVisible(bool(checked))

    def set_walls_message(self, text: str) -> None:
        """Put a sentence in the legend's slot -- the budget refusal.

        The same line the legend uses, because it answers the same
        question: the user ticked a box and something other than a wall
        picture happened.
        """
        legend = getattr(self, "split_walls_legend", None)
        if legend is None:
            return
        legend.setText(str(text or ""))
        legend.setVisible(bool(text))

    def _on_split_show_walls_toggled(self, checked: bool) -> None:
        """The tick moved by hand: swap the legend, then tell the
        orchestrator, which owns the layer."""
        legend = getattr(self, "split_walls_legend", None)
        if legend is not None:
            legend.setText(self.walls_legend_text() if checked else "")
            legend.setVisible(bool(checked))
        cb = getattr(self, "_on_show_walls_toggled_cb", None)
        if callable(cb):
            try:
                cb(bool(checked))
            except Exception as exc:  # noqa: BLE001
                print("[WALLS] toggle callback raised: {!r}".format(exc))

    #: What Accept's tooltip says when nothing is blocking it.
    _ACCEPT_DEFAULT_TIP = (
        "Write the ticked pieces into the volume as separate objects. "
        "Anything you unticked stays part of the original object -- it is "
        "never deleted."
    )

    def set_accept_reason(self, reason: str = "") -> None:
        """Say on Accept itself why it is grey.

        A disabled button with no explanation is the same bug as a
        silently dead one.  This is the TOOLTIP only -- unlike
        ``update_split_guided_ui(blocked_reason=...)`` it does not claim
        the status line or turn the dot amber, because leaving review is
        not a job running.
        """
        btn = getattr(self, "split_accept_btn", None)
        if btn is None:
            return
        btn.setToolTip(str(reason) or self._ACCEPT_DEFAULT_TIP)

    def clear_expected_count(self) -> None:
        """Un-arm "I expect exactly n objects", keeping the number.

        The ONE per-session preference that resets at a successful
        Accept.  Every other survivor -- the sources,
        the channels, the wall kind, the solver -- is an inconvenience
        if it carries over; this one is a silently WRONG cut on the next
        object.  The spin keeps its value, so re-arming it is one click
        and nothing the user chose is forgotten.
        """
        cb = getattr(self, "split_target_count_cb", None)
        if cb is None or not cb.isChecked():
            return
        cb.setChecked(False)
        self._refresh_split_guided_summary()
        print("[RESHAPE-LOOP] expected-count tick reset after a commit "
              "(the number is kept)")

    def _seed_swatch_icon(self, sid):
        """Small solid-colour icon matching seed *sid*'s colour in the views
        (napari's default Labels colormap, which the Refine layer uses), so the
        seed list reads as a colour legend.  Falls back to grey on any error."""
        from qtpy.QtGui import QColor

        color = QColor(204, 204, 204)
        try:
            import numpy as _np
            from napari.utils.colormaps import label_colormap

            if getattr(self, "_seed_swatch_cmap", None) is None:
                self._seed_swatch_cmap = label_colormap(49, seed=0.5)
            rgba = self._seed_swatch_cmap.map(
                _np.array([int(sid)], dtype=_np.intp)
            )[0]
            color = QColor(
                int(rgba[0] * 255), int(rgba[1] * 255),
                int(rgba[2] * 255), int(rgba[3] * 255),
            )
        except Exception:
            pass
        return self._colour_swatch_icon(color)

    @staticmethod
    def _colour_swatch_icon(color):
        """The swatch chip of one ``QColor`` (the seed list's and the Add
        channel items')."""
        # The mockup's `.sw`: a 12 px rounded chip.  Painted at 4x and
        # let the view scale it down, so the corners stay clean at the
        # 12 px icon size the ticklists ask for.
        from qtpy.QtCore import QRectF
        from qtpy.QtGui import QColor, QIcon, QPixmap
        from qtpy.QtGui import QPainter, QPainterPath

        pix = QPixmap(48, 48)
        pix.fill(QColor(0, 0, 0, 0))
        try:
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing, True)
            path = QPainterPath()
            path.addRoundedRect(QRectF(2, 2, 44, 44), 8, 8)
            painter.fillPath(path, color)
            painter.end()
        except Exception:
            pix = QPixmap(14, 14)
            pix.fill(color)
        return QIcon(pix)

    def update_split_ui_state(
        self,
        *,
        seed_count: int | None = None,
        seed_ids: list | None = None,
        active_seed_id: int | None = None,
        split_active: bool | None = None,
        start_enabled: bool | None = None,
        next_seed_enabled: bool | None = None,
        undo_seed_enabled: bool | None = None,
        refine_seeds_enabled: bool | None = None,
        preview_enabled: bool | None = None,
        accept_enabled: bool | None = None,
        reject_enabled: bool | None = None,
        reject_text: str | None = None,
        status_text: str | None = None,
        edt_enabled: bool | None = None,
        refine_further_enabled: bool | None = None,
        fragment_refine_active: bool | None = None,
        fragment_ids: list | None = None,
        active_fragment_id: int | None = None,
        detect_enabled: bool | None = None,
        detect_visible: bool | None = None,
        detect_histogram_data: dict | None = None,
        detect_threshold: float | None = None,
        detect_seed_count: int | None = None,
        # --- panel rework: the only two ADDITIONS ---
        review_active: bool | None = None,
        accept_text: str | None = None,
        # --- Brick 2a: the marker picker ---
        marker_ids: list | None = None,
        marker_stats: dict | None = None,
        marker_busy: bool | None = None,
        marker_note: str | None = None,
        # --- fix pass H2: a Find that has to spare SOME ticks ---------
        marker_preserve_ids: list | None = None,
        # --- markers taken from labels: the rows the touching rule
        # leaves unticked, as {marker id: contact}; the contact's
        # inside / object_voxels say why on the row's tooltip.
        marker_start_off: dict | None = None,
        # --- Brick U3: the wall-picture tick ------------
        walls_toggle_enabled: bool | None = None,
        # ...and WHY it is grey, when there is something worth saying
        # (fix pass H4/H5).  Qt shows no tooltip on a disabled widget,
        # so the reason is rendered into the legend's slot instead.
        #
        # `None`, not `""`: every keyword on these setters defaults to
        # None and `test_split_params_golden` freezes that, because the
        # setters are ADDITIVE and None is how "the caller said nothing"
        # is told apart from a value the caller meant.  It is normalised
        # to `""` at the one place it is used, so omitting it behaves
        # exactly as the `""` default did.
        walls_toggle_reason: str | None = None,
    ) -> None:
        """Update the split section UI from an external orchestrator.

        ``review_active`` drives the S5 swap and must be pushed from BOTH
        staging sites -- the manual path stages without ever calling
        ``update_split_guided_ui``, so inferring "review" from the piece
        count would silently fail it.  ``accept_text`` exists
        for fragment refine's "Accept all (n)", which otherwise inherits
        whatever the piece checklist last wrote.
        """
        # --- the marker picker ---------------------------
        if marker_ids is not None:
            self._populate_marker_list(
                marker_ids, marker_stats or {},
                preserve_ids=marker_preserve_ids,
                start_off=marker_start_off)
        if marker_busy is not None:
            self._set_marker_busy(bool(marker_busy))
        if marker_note is not None:
            self._set_marker_count_text(str(marker_note))

        if seed_count is not None and hasattr(self, "split_seed_label"):
            # The TICKED rows when there is a list; the pushed count when
            # there is none.
            _lst = getattr(self, "split_marker_list", None)
            if _lst is not None and _lst.count():
                self._set_seed_count_text(
                    len(self.get_checked_marker_ids() or []))
            else:
                self._set_seed_count_text(seed_count)
        # `seed_ids=` fills THE marker list now.  Every
        # existing orchestrator call site keeps working; what changes is
        # that painted markers land in the same tickable list a Find
        # fills, so every source can be cherry-picked.  A RECONCILE
        # preserves the ticks by id -- a Find does not, because the
        # detectors renumber from 1 on every run.
        if seed_ids is not None:
            # `marker_stats` rides along, because a reconcile can carry
            # rows a FIND produced (found markers plus a hand-painted
            # addition), and those keep their size and
            # position.  A purely painted row has no entry and so gets
            # no number, which is the rule.
            self._populate_marker_list(
                seed_ids, marker_stats or {}, preserve=True)
        # Highlight the marker being painted.  `blockSignals` is the
        # point: `setCurrentRow` emits `currentRowChanged`, and nothing
        # outside the panel may move because the panel re-rendered.
        if active_seed_id is not None and hasattr(self, "split_marker_list"):
            for row in range(self.split_marker_list.count()):
                item = self.split_marker_list.item(row)
                if item and item.data(Qt.UserRole) == active_seed_id:
                    self.split_marker_list.blockSignals(True)
                    self.split_marker_list.setCurrentRow(row)
                    self.split_marker_list.blockSignals(False)
                    break
        if split_active is not None:
            self._split_is_active = bool(split_active)
            self._apply_split_marker_list_visibility()
            # Two lines that are only true BEFORE Start; they cost 50 px
            # of a 247 px viewport once the session is running, which is
            # exactly the height S3 needs (real estate).
            for _attr in ("_split_target_hint", "split_about_btn",
                          "_split_about_line",
                          # MEASURED (Brick U2, 2026-09-20): leaving this
                          # grey line up during a session puts
                          # review+markers at 789 px against the 753 px
                          # ceiling UX-round 1.4 says may not be raised
                          # again without the owner.  Hidden, the same
                          # state is 752 px.  It goes with the target
                          # hint; the tooltip carries the same words for
                          # the whole session, so only the 37 px is lost.
                          # The TICK ITSELF stays up -- it is worth 13 px
                          # and it is exactly during a session, painting
                          # markers in the side views, that you want it.
                          "_split_contour_help"):
                _w = getattr(self, _attr, None)
                if _w is not None:
                    _w.setVisible(not split_active)
            # ...but NOT the body, whose rule is two-sided: it is on
            # screen only when the session is over AND the chevron is
            # open.  Putting it in the loop above hides it correctly at
            # Start and then pops all four paragraphs open the instant
            # the session ends, under a chevron the user never touched.
            self._refresh_split_about_body()
            if hasattr(self, "split_start_btn"):
                # The same object, relabelled: once the cell is held out of
                # the volume the only thing this button could mean is
                # "change which cell", and that is exactly what it may not
                # do.
                self.split_start_btn.setText(
                    "Change" if split_active else "Start"
                )
            self._refresh_split_stage()
        if walls_toggle_enabled is not None:
            self.set_walls_toggle_enabled(
                bool(walls_toggle_enabled), walls_toggle_reason or "")
        if review_active is not None:
            self._split_review_active = bool(review_active)
            if not review_active:
                # Brick U3: no result, no wall picture.  The layer is
                # dropped by the orchestrator at the same six exits;
                # this is the half that keeps SCREEN honest, so the two
                # can never disagree about whether it is up.
                self.clear_walls_toggle()
                # Mirrors the Add tab's swap, which clears its candidate
                # checklist on every transition (AFW _add_review_container):
                # a stale result must not sit under a new stage.
                self._clear_split_result_rows()
                if hasattr(self, "split_accept_btn"):
                    # `accept_text` ("Accept all (3)") is written by fragment
                    # refine and by nothing else; without this it leaked into
                    # the next session, and into a manual two-piece preview
                    # that then counted three.
                    self.split_accept_btn.setText("Accept")
            if hasattr(self, "split_preview_btn"):
                # "Preview again" -- the same button, re-solving the same
                # cell with whatever the user changed (#7).
                self.split_preview_btn.setText(
                    "Preview again" if review_active else "Preview"
                )
            self._refresh_split_stage()
        if start_enabled is not None and hasattr(self, "split_start_btn"):
            self.split_start_btn.setEnabled(start_enabled)
        if next_seed_enabled is not None and hasattr(self, "split_next_seed_btn"):
            self.split_next_seed_btn.setEnabled(next_seed_enabled)
        if undo_seed_enabled is not None and hasattr(self, "split_undo_seed_btn"):
            self.split_undo_seed_btn.setEnabled(undo_seed_enabled)
        if refine_seeds_enabled is not None and hasattr(self, "split_refine_seeds_btn"):
            self.split_refine_seeds_btn.setEnabled(refine_seeds_enabled)
            # "Edit markers" means "put the preview away and paint some
            # more", so it has nothing to do before there IS a preview.
            # On screen and greyed it is a 28 px row of a 243 px screen
            # in the state that gained the whole painting block this
            # round (real estate).
            self.split_refine_seeds_btn.setVisible(bool(refine_seeds_enabled))
        if preview_enabled is not None and hasattr(self, "split_preview_btn"):
            self.split_preview_btn.setEnabled(preview_enabled)
        if accept_enabled is not None and hasattr(self, "split_accept_btn"):
            self.split_accept_btn.setEnabled(accept_enabled)
            if accept_enabled:
                # Whatever reason it was carrying is over -- on the
                # button AND under it (fix pass H3).
                self.set_accept_reason("")
                self.set_footer_reason("")
        if reject_enabled is not None and hasattr(self, "split_reject_btn"):
            self.split_reject_btn.setEnabled(reject_enabled)
        if accept_text is not None and hasattr(self, "split_accept_btn"):
            self.split_accept_btn.setText(accept_text)
        if reject_text is not None and hasattr(self, "split_reject_btn"):
            self.split_reject_btn.setText(reject_text)
            self._refresh_split_footer_summary()
        if status_text is not None and hasattr(self, "split_status_label"):
            self.split_status_label.setText(status_text)
        if edt_enabled is not None and hasattr(self, "split_edt_cb"):
            self.split_edt_cb.setEnabled(edt_enabled)
            # Auto-uncheck when disabled (split ended).
            if not edt_enabled and self.split_edt_cb.isChecked():
                self.split_edt_cb.setChecked(False)

        # --- Detect Seeds UI ---
        if detect_enabled is not None and hasattr(self, "split_detect_seeds_btn"):
            self.split_detect_seeds_btn.setEnabled(detect_enabled)
        if detect_visible is not None and hasattr(self, "_detect_container"):
            self._detect_container.setVisible(detect_visible)
        if detect_histogram_data is not None:
            self._update_detect_histogram(detect_histogram_data)
        if detect_threshold is not None:
            self._set_detect_threshold(detect_threshold)
        if detect_seed_count is not None and hasattr(self, "_detect_count_label"):
            self._detect_count_label.setText(
                "Found: {} marker{}".format(
                    detect_seed_count,
                    "" if detect_seed_count == 1 else "s"))

        # --- Fragment Refine ---
        if refine_further_enabled is not None and hasattr(self, "split_refine_further_btn"):
            self.split_refine_further_btn.setEnabled(refine_further_enabled)
            # S5 shows only what there is to show, so the hand-off button is
            # absent rather than greyed until a result exists.
            self.split_refine_further_btn.setVisible(bool(refine_further_enabled))
        if fragment_refine_active is not None:
            # U4: the fourth stage flag.  S3 SOURCES and the
            # whole marker-work block are locked while it is set.
            self._split_fragment_active = bool(fragment_refine_active)
            self._refresh_split_stage()
        if fragment_refine_active is not None and hasattr(self, "split_fragment_list"):
            self.split_fragment_list.setVisible(fragment_refine_active)
            if not fragment_refine_active:
                self.split_fragment_list.clear()
        if fragment_ids is not None and hasattr(self, "split_fragment_list"):
            self.split_fragment_list.blockSignals(True)
            self.split_fragment_list.clear()
            for fid in fragment_ids:
                item = QListWidgetItem(f"Fragment {fid}")
                item.setData(Qt.UserRole, fid)
                self.split_fragment_list.addItem(item)
            self.split_fragment_list.blockSignals(False)
        if active_fragment_id is not None and hasattr(self, "split_fragment_list"):
            for i in range(self.split_fragment_list.count()):
                item = self.split_fragment_list.item(i)
                if item and item.data(Qt.UserRole) == active_fragment_id:
                    self.split_fragment_list.blockSignals(True)
                    self.split_fragment_list.setCurrentItem(item)
                    self.split_fragment_list.blockSignals(False)
                    break

        if any(
            v is not None
            for v in (refine_further_enabled, fragment_refine_active,
                      fragment_ids)
        ):
            self._refresh_split_result_visibility()

    def _populate_marker_list(self, marker_ids, marker_stats,
                              preserve: bool = False,
                              preserve_ids=None, start_off=None) -> None:
        """Fill the one marker list.

        Every marker starts TICKED: a marker the detector found is an
        object until the user says otherwise.  There is no sliver rule
        here -- that is about pieces, which have a size distribution; a
        marker is one voxel blob and small does not mean spurious.

        ``preserve`` is NARROW and deliberately so.  A RECONCILE (the
        orchestrator re-listing what is painted on the Refine layer)
        keeps the ticks by id.  A completed FIND does not: the detectors
        number the markers by enumeration order over the crop, so they
        restart from 1 on every run, and "id 2 was unticked" would
        untick a different marker.

        ``preserve_ids`` is the NARROWER form of the same thing, and it
        is what a materialising source needs (fix pass H2): source 1's
        Find re-ticks the markers IT found -- their ids restart -- but
        the hand-painted ones beside them keep their ids, and re-ticking
        a marker the user deliberately unticked is the panel undoing a
        decision behind their back.  Only the named ids may keep an OFF
        state; everything else is ticked.

        ``start_off`` is a Find on markers taken from labels: the labels
        that only touch the object start UNTICKED, with the word
        "touching" on the row and the reason on its tooltip.  The
        verdicts stay with their rows through a reconcile; a Find that
        brings none forgets them.
        """
        lst = getattr(self, "split_marker_list", None)
        if lst is None:
            return
        stats = {int(k): v for k, v in (marker_stats or {}).items()}
        wanted = {int(m) for m in (marker_ids or ())}
        if start_off is not None:
            self._marker_touching = {
                int(k): v for k, v in dict(start_off).items()}
        elif not preserve:
            self._marker_touching = {}
        touching = {
            k: v for k, v in (
                getattr(self, "_marker_touching", None) or {}).items()
            if k in wanted
        }
        self._marker_touching = dict(touching)
        start_unticked = set(touching) if start_off is not None else set()
        spare = (None if preserve_ids is None
                 else {int(i) for i in preserve_ids})
        was_off = set()
        if preserve or spare:
            for i in range(lst.count()):
                it = lst.item(i)
                if it is None or it.checkState() == Qt.Checked:
                    continue
                mid = it.data(Qt.UserRole)
                if mid is None:
                    continue
                mid = int(mid)
                if preserve or mid in spare:
                    was_off.add(mid)
        lst.blockSignals(True)
        lst.clear()
        self._marker_tick_item = None
        for i, mid in enumerate(marker_ids or ()):
            mid = int(mid)
            voxels, centre = self._marker_stat(stats.get(mid))
            text = "Marker {}".format(i + 1)
            tip = "Marker {} (id {})".format(i + 1, mid)
            if voxels is not None:
                tip += " -- {:,} voxel{}".format(
                    int(voxels), "" if int(voxels) == 1 else "s")
            if centre is not None:
                tip += " -- at {}".format(self._marker_place(centre))
            # A SIZE only earns the row when it says something.  The
            # detectors return single-voxel peaks, so "1" for every marker
            # is a column that distinguishes nothing; where the marker IS
            # does -- it is how you find it in the viewer.
            if voxels is not None and int(voxels) > 1:
                text += "  {}  {:,}".format(_DOT_SEP, int(voxels))
            elif centre is not None:
                text += "  {}  {}".format(_DOT_SEP, self._marker_place(centre))
            if mid in touching:
                text += "  {}  {}".format(_DOT_SEP, _RT.MARKER_ROW_TOUCHING)
                tip += "\n" + self._touching_tip(touching[mid])
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, mid)
            try:
                item.setIcon(self._seed_swatch_icon(mid))
            except Exception:
                pass
            item.setToolTip(tip)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Unchecked if (mid in was_off or mid in start_unticked)
                else Qt.Checked)
            self._style_marker_row(item)
            lst.addItem(item)
        lst.blockSignals(False)
        self._fit_marker_list()
        self._apply_split_marker_list_visibility()
        self._refresh_marker_count_line()
        self._refresh_split_guided_summary()
        print("[MARKERS] list now holds {} marker(s), {} kept from before"
              .format(lst.count(), len(was_off)))

    @staticmethod
    def _touching_tip(contact) -> str:
        """The touching row's tooltip, from a contact dict or object."""
        def _get(name):
            if isinstance(contact, dict):
                return contact.get(name, 0)
            return getattr(contact, name, 0)

        try:
            n = int(_get("inside") or 0)
            obj = int(_get("object_voxels") or 0)
        except Exception:
            n, obj = 0, 0
        share = 100.0 * n / obj if obj > 0 else 0.0
        # One voxel of a big object rounds to "0.0", which reads as none.
        pct = ("under 0.1" if 0.0 < share < 0.05
               else "{:.1f}".format(share))
        return _RT.MARKER_TOUCHING_TIP.format(n=n, s=_RT.plural(n), pct=pct)

    @staticmethod
    def _marker_stat(value):
        """``(voxels, centroid_zyx)`` from whatever the caller sent.

        A bare int is the original shape (id -> voxels) and still works;
        a mapping may carry ``voxels`` and ``centroid_zyx``.
        """
        if value is None:
            return None, None
        if isinstance(value, dict):
            voxels = value.get("voxels")
            centre = value.get("centroid_zyx")
            try:
                voxels = None if voxels is None else int(voxels)
            except Exception:
                voxels = None
            if centre is not None:
                try:
                    centre = tuple(float(v) for v in centre)[:3]
                    if len(centre) != 3:
                        centre = None
                except Exception:
                    centre = None
            return voxels, centre
        try:
            return int(value), None
        except Exception:
            return None, None

    @staticmethod
    def _marker_place(centre) -> str:
        """``z 12 y 240 x 89`` -- whole voxels, because a centroid of
        240.4 is not a place anyone can navigate to."""
        try:
            z, y, x = (int(round(float(v))) for v in centre)
        except Exception:
            return ""
        return "z {} y {} x {}".format(z, y, x)

    def _set_marker_busy(self, busy: bool) -> None:
        """A Find is running.  S3 stays OPEN.

        The solve's busy state collapses SOURCES, which is right for a
        solve and absurd for a Find: it would hide the button that
        started it and the list it is about to fill.  So this is a LOCAL
        state -- the Find button and the marker knobs go grey, Preview
        goes grey with Stop beside it and the bar runs -- and it never
        touches ``_split_is_busy``.
        """
        busy = bool(busy)
        btn = getattr(self, "split_find_markers_btn", None)
        if btn is not None:
            btn.setEnabled(not busy)
        # ...and the LIST itself (U2b, audit L1).  A tick used to be
        # only a filter the Preview reads, so leaving it live during a
        # Find was harmless.  Now a tick MOVES VOXELS on the layer the
        # Find is about to write its own markers onto, which is the
        # same reason the solve locks it -- so the two locks match.
        for attr in ("split_marker_list", "split_marker_all_btn",
                     "split_marker_none_btn"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.setEnabled(not busy)
                except Exception:
                    pass
        for attr in self._MARKER_KNOBS:
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.setEnabled(not busy)
                except Exception:
                    pass
        # The bar and the Stop button are SHARED with the solve.  A Find
        # that releases them while a solve is running would take away
        # the only way to stop the solve, so it only ever hides them
        # when nothing else owns them.
        solve_owns = bool(getattr(self, "_split_is_busy", False))
        for attr in ("_split_busy_container", "split_cancel_preview_btn"):
            w = getattr(self, attr, None)
            if w is None:
                continue
            if busy:
                w.setVisible(True)
            elif not solve_owns:
                w.setVisible(False)
        pv = getattr(self, "split_preview_btn", None)
        if pv is not None and busy:
            pv.setEnabled(False)
        if busy:
            self._set_marker_count_text("Finding...")
            bar = getattr(self, "split_busy_bar", None)
            if bar is not None:
                bar.setRange(0, 0)

    def _ensure_detect_plot(self) -> None:
        """Lazily create the pyqtgraph histogram widget for auto-detect.

        Sets ``_detect_plot_widget`` only after full initialization succeeds
        so a partial failure (e.g. Qt enum mismatch) doesn't leave a stale
        reference that prevents retry on the next call.
        """
        if self._detect_plot_widget is not None:
            # Already initialized — but check it was actually inserted.
            if self._detect_plot_placeholder is None:
                return
            # Partial init from a previous error — fall through to retry.
            self._detect_plot_widget = None
        if self._detect_plot_placeholder is None:
            # released for a panel being torn down: nowhere to put a plot
            # (built anyway it would be parentless, a window of its own)
            return

        try:
            import pyqtgraph as pg
        except ImportError:
            return

        pw = pg.PlotWidget()
        pw.setBackground("#1e1e2e")
        pw.setFixedHeight(110)
        pw.getPlotItem().hideButtons()
        pw.setMouseEnabled(x=False, y=False)

        # Axis labels — style dict is passed directly to setLabel.
        axis_style = {"color": "#aaa", "font-size": "9px"}
        # Not "EDT value": the axis is read by someone deciding where to
        # put the threshold line, and "how tall is this peak" is the
        # question they are answering.
        pw.setLabel("bottom", "Peak height (voxels)", **axis_style)
        pw.setLabel("left", "Count", **axis_style)
        pw.getAxis("bottom").setTextPen("#aaa")
        pw.getAxis("left").setTextPen("#aaa")

        # Draggable threshold line.
        line = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen("r", width=2, style=Qt.DashLine),
        )
        pw.addItem(line)
        line.sigPositionChanged.connect(self._on_detect_line_moved)

        # Replace placeholder with the live plot widget.
        placeholder = self._detect_plot_placeholder
        if placeholder is not None:
            parent_layout = placeholder.parentWidget().layout()
            if parent_layout is not None:
                idx = parent_layout.indexOf(placeholder)
                parent_layout.removeWidget(placeholder)
                placeholder.deleteLater()
                parent_layout.insertWidget(idx, pw)
            self._detect_plot_placeholder = None

        # Set instance refs only after everything succeeded.
        self._detect_plot_widget = pw
        self._detect_bar_item = None
        self._detect_thresh_line = line

    def _on_detect_line_moved(self, line) -> None:
        """Fire threshold callback when user drags the red line."""
        cb = getattr(self, "_on_threshold_changed_cb", None)
        if cb is not None:
            try:
                cb(float(line.value()))
            except Exception:
                pass

    def _update_detect_histogram(self, data: dict) -> None:
        """Redraw the EDT histogram from detect data.

        Parameters
        ----------
        data : dict
            ``{edt_values: list[float], n_bins: int}``
        """
        self._ensure_detect_plot()
        if self._detect_plot_widget is None:
            return

        try:
            import pyqtgraph as pg
        except ImportError:
            return

        import numpy as _np

        edt_values = _np.array(data.get("edt_values", []), dtype=float)
        self._detect_last_edt_values = list(edt_values)
        n_bins = data.get("n_bins", 20)

        pw = self._detect_plot_widget

        # Remove old bar item.
        if self._detect_bar_item is not None:
            pw.removeItem(self._detect_bar_item)
            self._detect_bar_item = None

        if len(edt_values) == 0:
            return

        counts, edges = _np.histogram(edt_values, bins=n_bins)
        widths = _np.diff(edges)
        bar = pg.BarGraphItem(
            x=edges[:-1], height=counts, width=widths * 0.9,
            brush="#4c78a8", pen=None,
        )
        pw.addItem(bar)
        self._detect_bar_item = bar
        pw.setXRange(float(edges[0]), float(edges[-1]), padding=0.05)

    def _set_detect_threshold(self, value: float) -> None:
        """Set threshold line position without firing the callback.  Never
        BUILDS the plot (only ``_update_detect_histogram`` does)."""
        if self._detect_thresh_line is not None:
            self._detect_thresh_line.blockSignals(True)
            self._detect_thresh_line.setValue(value)
            self._detect_thresh_line.blockSignals(False)

    def _on_bins_changed(self, n_bins: int) -> None:
        """Redraw histogram with the new bin count (live update)."""
        if not self._detect_last_edt_values:
            return
        self._update_detect_histogram({
            "edt_values": self._detect_last_edt_values,
            "n_bins": n_bins,
        })

    # --- Add (de-novo segmentation) section -----------------------------------

    def _build_add_denovo_section(self) -> QWidget:
        """Build the Add page.

        Fill cleared space inside a box with many new objects: two presets
        (**Nuclei** -- brightness alone; **Cells with a membrane stain** --
        markers + walls through the engine's region target).  The six-slot
        skeleton of the Refine page: a collapsed description, S1 status,
        S2 "The box", S3 "What are you adding?" (the Brightness card, or
        the Markers / Walls chevrons, one open at a time), S5 the result
        card, Advanced at the bottom of the scroll, S4 PINNED between the
        scroll and the footer, S6 the footer.  Everything that switches with
        the preset, the source or the state is a hidden widget, never a
        ``QStackedWidget``.  REVIEW replaces S2/S3/S5/Advanced in place.
        """
        from packages.epicycle_annotator import add_preview as AP

        page, layout, footer, scroll = self._page_with_footer()
        content = scroll.widget()
        self._add_touched = set()
        self._add_applying_preset = False
        self._add_source_note = ""
        self._add_notes_full = ""
        self._add_candidate_small_ids = set()
        self._add_candidate_flags = {}
        self._add_footer_override = None
        self._add_busy_state = False
        self._add_source_keys = []
        # the Markers ▸ From list as the orchestrator gave it; on a Cells
        # whole patch the combo offers only its "Objects from ..." items and
        # Click markers, and the box's own pick waits here until the patch
        # is a box again
        self._add_source_entries_all = []
        self._add_sources_offered_whole = False
        self._add_source_box_key = None
        self._add_nuclei_auto_value = None
        # the two Auto lines: being measured, or measured flat (nothing for
        # Otsu to split) -- the text is written by _apply_add_visibility
        self._add_brightness_pending = False
        self._add_brightness_flat = False
        self._add_nuclei_pending = False
        self._add_nuclei_flat = False
        self._add_run_count = None
        self._add_ui = {"box": False, "review": False, "candidates": False,
                        "default_spacing": False, "single_channel": False}

        _hdr = QHBoxLayout()
        _hdr_label = QLabel("Add objects")
        _hdr_label.setStyleSheet(SECTION_HEADER_STYLE)
        _hdr.addWidget(_hdr_label)
        _hdr.addStretch(1)
        _hdr.addWidget(_help_button(
            content,
            guide_path=str(_GUIDES_DIR / "annotator_3d_add.md"),
            title="Add objects -- 3D Annotator",
            tooltip="Help: Add",
        ))
        layout.addLayout(_hdr)

        # -- the collapsed description --------------------------
        _about_row = QHBoxLayout()
        _about_row.setContentsMargins(0, 0, 0, 0)
        _about_row.setSpacing(4)
        self.add_about_btn = QPushButton(_CHEVRON_CLOSED, content)
        self.add_about_btn.setCheckable(True)
        self.add_about_btn.setStyleSheet(PANEL_CHEVRON_STYLE)
        self.add_about_btn.setToolTip("What this tool does, in plain words.")
        self.add_about_btn.setFixedWidth(_about_btn_width(self.add_about_btn))
        _about_row.addWidget(self.add_about_btn, 0, Qt.AlignTop)
        self._add_about_line = _ClickableLabel(self.add_about_btn.toggle,
                                               content)
        self._add_about_line.setText(AP.ADD_ABOUT_LINE)
        self._add_about_line.setStyleSheet(
            PANEL_HELP_STYLE + _ClickableLabel.HOVER_STYLE)
        self._add_about_line.setWordWrap(True)
        _about_row.addWidget(self._add_about_line, 1)
        layout.addLayout(_about_row)
        self._add_about_body = QLabel(content)
        self._add_about_body.setTextFormat(Qt.RichText)
        self._add_about_body.setText(AP.ADD_ABOUT_BODY)
        self._add_about_body.setStyleSheet(PANEL_HELP_STYLE)
        self._add_about_body.setWordWrap(True)
        self._add_about_body.setVisible(False)
        layout.addWidget(self._add_about_body)

        def _about(checked):
            self._add_about_body.setVisible(bool(checked))
            self.add_about_btn.setText(
                _CHEVRON_OPEN if checked else _CHEVRON_CLOSED)
        self.add_about_btn.toggled.connect(_about)

        # ==================================================================
        # S1 STATUS
        # ==================================================================
        self.add_status_label = QLabel(AP.ADD_S1_IDLE)
        self.add_status_label.setStyleSheet(ADD_S1_STYLE)
        self.add_status_label.setWordWrap(True)
        self._add_s1_container = self._status_row(
            "_add_status_dot", self.add_status_label)
        layout.addWidget(self._add_s1_container)
        # Add at the default spacing runs (every path), and
        # says so in amber right under S1 -- never a refusal, never grey.
        self.add_spacing_note = QLabel(AP.ADD_NOTE_DEFAULT_SPACING)
        self.add_spacing_note.setWordWrap(True)
        self.add_spacing_note.setStyleSheet(PANEL_WARN_STYLE)
        self.add_spacing_note.setVisible(False)
        layout.addWidget(self.add_spacing_note)
        # D-1 (the owner's live test 2026-09-27): "Treat dark space as
        # outside" OFF on Cells lets the fill run to the box faces -- the
        # same amber slot and the same rules as the spacing note
        self.add_dark_off_note = QLabel(AP.ADD_NOTE_DARK_OFF_LEAK)
        self.add_dark_off_note.setWordWrap(True)
        self.add_dark_off_note.setStyleSheet(PANEL_WARN_STYLE)
        self.add_dark_off_note.setVisible(False)
        layout.addWidget(self.add_dark_off_note)

        # Everything but REVIEW lives in TUNE (S2 / S3 / S5 / Advanced).
        self._add_tune_container = QWidget()
        tune = QVBoxLayout(self._add_tune_container)
        tune.setContentsMargins(0, 0, 0, 0)
        tune.setSpacing(2)

        # ==================================================================
        # S2 THE BOX
        # ==================================================================
        self._add_s2_container = QWidget()
        _s2 = QVBoxLayout(self._add_s2_container)
        make_panel_card(self._add_s2_container)
        _s2.addWidget(panel_card_header("The box", self._add_s2_container))
        # no box: ONE row, [Place box] [Whole patch], both always enabled
        # -- the whole patch is a box size, never a grey door; a choice that
        # cannot fill a whole patch greys Preview and S1 says why
        self._add_idle_row = QWidget(self._add_s2_container)
        _irow = QHBoxLayout(self._add_idle_row)
        _irow.setContentsMargins(0, 0, 0, 0)
        _irow.setSpacing(4)
        self.add_start_btn = QPushButton("Place box", self._add_idle_row)
        self.add_start_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_start_btn.setToolTip(AP.ADD_HELP_BOX_EMPTY)
        _irow.addWidget(self.add_start_btn, 1)
        self.add_whole_btn = QPushButton(AP.ADD_WHOLE_BTN, self._add_idle_row)
        self.add_whole_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_whole_btn.setToolTip(AP.ADD_WHOLE_TIP)
        _irow.addWidget(self.add_whole_btn, 1)
        _s2.addWidget(self._add_idle_row)

        self._add_box_container = QWidget()
        _brow = QHBoxLayout(self._add_box_container)
        _brow.setContentsMargins(0, 0, 0, 0)
        _brow.setSpacing(3)
        # "Size" before S M L, as the mockup (B2 fix pass U-M5d).  To fit
        # the 290 px inside the card, S / M / L are 26 px and Numbers... /
        # Recenter carry 5 px of side padding, not 10.
        self._add_size_label = QLabel("Size", self._add_box_container)
        self._add_size_label.setStyleSheet(PANEL_ROW_LABEL_STYLE)
        _brow.addWidget(self._add_size_label)
        # Live-shot fix V8: S / M / L are sized to their own hints now
        # (readable letters win), and the row no longer fits with "Size"
        # in front -- offscreen 301 px with it, 269 without, against the
        # 286 px card; live 271 / 241.  So the label comes OFF the row
        # (constructed, hidden;  records it), never a letter.
        self._add_size_label.setVisible(False)
        for key, frac, tip in (
            ("S", 0.125, "About an eighth of the patch on each side."),
            ("M", 0.25, "About a quarter of the patch on each side."),
            ("L", 0.5, "About half of the patch on each side (cells: at most "
                       "{} voxels).".format(_AP_voxel_limit_words(
                           _AP_ADD_CELLS_PRESET_MAX_VOXELS))),
        ):
            b = QPushButton(key)
            b.setStyleSheet(PANEL_BTN_STYLE)
            # sized to its own hint: a fixed 26 px clipped the letters live
            # (hints 29 / 33 / 28 px; "M" drew as a "V") -- live-shot V8
            b.setMinimumWidth(b.sizeHint().width())
            b.setToolTip(tip)
            b.clicked.connect(
                lambda _c=False, f=frac: self._fire_add_size_preset(f))
            setattr(self, "add_size_{}_btn".format(key.lower()), b)
            _brow.addWidget(b)
        # the fourth size: the whole patch (only S, M, L or a Numbers...
        # slider make it a box again).  It takes the room the word
        # "Recenter" gave up, so the row is no wider than it was
        self.add_size_whole_btn = QPushButton(AP.ADD_WHOLE_SIZE_BTN)
        self.add_size_whole_btn.setStyleSheet(_ADD_TIGHT_BTN_STYLE)
        self.add_size_whole_btn.setToolTip(AP.ADD_WHOLE_TIP)
        _brow.addWidget(self.add_size_whole_btn)
        self.add_numbers_btn = QPushButton("Numbers…")
        self.add_numbers_btn.setStyleSheet(_ADD_TIGHT_BTN_STYLE)
        self.add_numbers_btn.setToolTip(
            "Type the box's centre and size, one number per axis.")
        _brow.addWidget(self.add_numbers_btn)
        # Recenter: its glyph alone, the words in its tooltip
        self.add_recenter_btn = QPushButton("⌖")
        self.add_recenter_btn.setStyleSheet(_ADD_TIGHT_BTN_STYLE)
        self.add_recenter_btn.setToolTip(
            "Recenter: move the box so its centre is where the crosshair is.")
        _brow.addWidget(self.add_recenter_btn)
        _brow.addStretch(1)
        self._add_box_container.setVisible(False)
        _s2.addWidget(self._add_box_container)
        self._add_box_help = QLabel(AP.ADD_HELP_BOX_EMPTY)
        self._add_box_help.setWordWrap(True)
        self._add_box_help.setStyleSheet(_HELPER_LINE_STYLE)
        _s2.addWidget(self._add_box_help)
        tune.addWidget(self._add_s2_container)

        # Numbers... -- the six sliders, re-homed in a non-modal dialog
        self._add_numbers_dialog = _AddBoxNumbers(self)
        self._add_box_sliders = self._add_numbers_dialog.sliders
        for sld in self._add_box_sliders.values():
            sld.valueChanged.connect(self._on_add_box_slider_changed)
        self.add_numbers_btn.clicked.connect(self._open_add_numbers)
        # the old separator stays constructed (never placed) for callers
        self._add_box_sep = QLabel("", self)
        self._add_box_sep.setVisible(False)

        # ==================================================================
        # S3 WHAT ARE YOU ADDING?
        # ==================================================================
        self._add_s3_container = QWidget()
        _s3 = QVBoxLayout(self._add_s3_container)
        make_panel_card(self._add_s3_container)
        _s3.addWidget(panel_card_header("What are you adding?",
                                        self._add_s3_container))
        _pr = QHBoxLayout()
        _pr.setSpacing(6)
        _pr.addWidget(panel_form_label("Adding", parent=self))
        self.add_preset_combo = QComboBox()
        self.add_preset_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.add_preset_combo.addItems(list(AP.ADD_PRESET_LABELS))
        for _i, _tip in enumerate(AP.ADD_PRESET_TIPS):
            self.add_preset_combo.setItemData(_i, _tip, Qt.ToolTipRole)
        _pr.addWidget(self.add_preset_combo, 1)
        _s3.addLayout(_pr)
        self._add_preset_help = QLabel(AP.ADD_HELP_NUCLEI)
        self._add_preset_help.setWordWrap(True)
        self._add_preset_help.setStyleSheet(_HELPER_LINE_STYLE)
        _s3.addWidget(self._add_preset_help)

        # ---- Brightness (Nuclei) ----
        self._add_brightness_container = QWidget()
        _bc = QVBoxLayout(self._add_brightness_container)
        _bc.setContentsMargins(0, 0, 0, 0)
        _bc.setSpacing(2)
        _bc.addWidget(panel_card_header("Brightness",
                                        self._add_brightness_container))
        self.add_channel_combo = QComboBox()
        self.add_channel_combo.setToolTip(
            "The image channel the nuclei are bright in.")
        _bc.addLayout(self._refine_row("Channel", self.add_channel_combo))
        # "Look for nuclei" directly under Channel: Anywhere (index 0, the
        # brightness finder as it always was) or Inside each cell of another
        # run.  Read BY INDEX through ``_add_look_keys``; a run is listed
        # but never pre-selected.
        self._add_look_keys = [(AP.LOOK_ANYWHERE, None)]
        self.add_look_for_combo = QComboBox()
        self.add_look_for_combo.addItem(AP.ADD_LOOK_ANYWHERE)
        self.add_look_for_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.add_look_for_combo.setMinimumContentsLength(10)
        self._add_look_row = QWidget(self._add_brightness_container)
        _lr = self._refine_row(AP.ADD_LOOK_FOR_LABEL, self.add_look_for_combo)
        _lr.setContentsMargins(0, 0, 0, 0)
        self._add_look_row.setLayout(_lr)
        _bc.addWidget(self._add_look_row)
        # Inside: what it does (or why it cannot), and the run's cells in
        # the box -- shown before a result only
        self._add_inside_help = QLabel(AP.ADD_HELP_INSIDE)
        self._add_inside_help.setWordWrap(True)
        self._add_inside_help.setStyleSheet(_HELPER_LINE_STYLE)
        self._add_inside_help.setVisible(False)
        _bc.addWidget(self._add_inside_help)
        self._add_inside_count = QLabel(AP.ADD_INSIDE_COUNT_PENDING)
        self._add_inside_count.setWordWrap(True)
        self._add_inside_count.setStyleSheet(_HELPER_LINE_STYLE)
        self._add_inside_count.setVisible(False)
        _bc.addWidget(self._add_inside_count)
        self.add_clahe_cb = QCheckBox("Local contrast")
        self.add_clahe_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_clahe_cb.setToolTip(
            "Evens out bright and dim regions of the box before the "
            "threshold. Off by default.")
        _bc.addWidget(self.add_clahe_cb)
        self._add_plot_widget = None
        self._add_bar_item = None
        self._add_thresh_line = None
        self._add_last_values = []
        self._add_plot_placeholder = QWidget()
        self._add_plot_placeholder.setFixedHeight(100)
        _bc.addWidget(self._add_plot_placeholder)
        _thr = QHBoxLayout()
        _thr.setSpacing(6)
        _thr.addWidget(panel_form_label("Threshold", parent=self))
        self.add_threshold_sb = QDoubleSpinBox()
        self.add_threshold_sb.setDecimals(3)
        self.add_threshold_sb.setRange(0.0, 1.0)
        self.add_threshold_sb.setSingleStep(0.01)
        self.add_threshold_sb.setKeyboardTracking(False)
        self.add_threshold_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.add_threshold_sb.setToolTip(AP.ADD_HELP_THRESHOLD)
        self.add_threshold_sb.setMinimumWidth(
            self._readable_spin_width(self.add_threshold_sb))
        _thr.addWidget(self.add_threshold_sb, 1)
        self.add_otsu_btn = QPushButton("Auto")
        self.add_otsu_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_otsu_btn.setToolTip(
            "Put the threshold at the automatic split between background "
            "and the bright objects (Otsu).")
        _thr.addWidget(self.add_otsu_btn)
        # a row widget, so Inside each cell can hide it (no threshold there)
        _thr.setContentsMargins(0, 0, 0, 0)
        self._add_threshold_row = QWidget(self._add_brightness_container)
        self._add_threshold_row.setLayout(_thr)
        _bc.addWidget(self._add_threshold_row)
        self._add_threshold_help = QLabel(AP.ADD_HELP_THRESHOLD)
        self._add_threshold_help.setWordWrap(True)
        self._add_threshold_help.setStyleSheet(_HELPER_LINE_STYLE)
        _bc.addWidget(self._add_threshold_help)
        self.add_threshold_sb.editingFinished.connect(
            self._on_add_threshold_typed)
        _s3.addWidget(self._add_brightness_container)

        # ---- Cells: Markers + Walls (one open at a time) ----
        self._add_cells_container = QWidget()
        _cc = QVBoxLayout(self._add_cells_container)
        _cc.setContentsMargins(0, 0, 0, 0)
        _cc.setSpacing(2)
        self.add_markers_btn = self._subsection_chevron(
            "Markers", "Where each new object grows from.")
        self.add_markers_btn.setProperty("_epicycle_elide_title", True)
        _cc.addWidget(self.add_markers_btn)
        self._add_markers_container = QWidget()
        _mk = QVBoxLayout(self._add_markers_container)
        _mk.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _mk.setSpacing(2)
        # the amber membrane-only line sits directly under the Markers
        # summary, above the source combo (mockup 2b; U-M5a)
        self._add_membrane_only_label = QLabel(AP.ADD_AMBER_MEMBRANE_ONLY)
        self._add_membrane_only_label.setWordWrap(True)
        self._add_membrane_only_label.setStyleSheet(PANEL_WARN_STYLE)
        self._add_membrane_only_label.setVisible(False)
        _mk.addWidget(self._add_membrane_only_label)
        self.add_marker_source_combo = QComboBox()
        self.add_marker_source_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.add_marker_source_combo.setToolTip(
            "Where the markers come from: another run's objects, nuclei in "
            "a second channel, or your clicks.")
        self.add_marker_source_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.add_marker_source_combo.setMinimumContentsLength(10)
        # a row widget (no margins: the same height as the bare row), so
        # the Nuclei preset can hide it -- its markers are the clicks
        self._add_source_row = QWidget(self._add_markers_container)
        _sr = self._refine_row("From", self.add_marker_source_combo)
        _sr.setContentsMargins(0, 0, 0, 0)
        self._add_source_row.setLayout(_sr)
        _mk.addWidget(self._add_source_row)
        self._add_source_help = QLabel("")
        self._add_source_help.setWordWrap(True)
        self._add_source_help.setStyleSheet(_HELPER_LINE_STYLE)
        # nuclei rows
        self._add_nuclei_rows = QWidget()
        _nr = QVBoxLayout(self._add_nuclei_rows)
        _nr.setContentsMargins(0, 0, 0, 0)
        _nr.setSpacing(2)
        self.add_nuclei_channel_combo = QComboBox()
        self.add_nuclei_channel_combo.setToolTip(
            "The channel the nuclei are bright in.")
        _nr.addLayout(self._refine_row("Nuclei channel",
                                       self.add_nuclei_channel_combo))
        _nt = QHBoxLayout()
        _nt.setSpacing(6)
        _nt.addWidget(panel_form_label("Threshold", parent=self))
        self.add_nuclei_threshold_sb = QDoubleSpinBox()
        self.add_nuclei_threshold_sb.setDecimals(3)
        self.add_nuclei_threshold_sb.setKeyboardTracking(False)
        self.add_nuclei_threshold_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.add_nuclei_threshold_sb.setToolTip(
            "The brightness above which a voxel counts as nucleus.")
        # the readable floor is sized for "65535.0" -- as wide as a 4-digit
        # value at 3 dp ("1234.567"); the 1e6 range's own width set the
        # page's minimum wider than the 316 px viewport (B2 fix pass U-H1)
        self.add_nuclei_threshold_sb.setRange(0.0, 65535.0)
        self.add_nuclei_threshold_sb.setDecimals(1)
        _floor = self._readable_spin_width(self.add_nuclei_threshold_sb)
        self.add_nuclei_threshold_sb.setProperty("_epicycle_readable_floor",
                                                 int(_floor))
        self.add_nuclei_threshold_sb.setMinimumWidth(_floor)
        self.add_nuclei_threshold_sb.setDecimals(3)
        self.add_nuclei_threshold_sb.setRange(0.0, 1.0e6)
        _nt.addWidget(self.add_nuclei_threshold_sb, 1)
        self.add_nuclei_auto_btn = QPushButton("Auto")
        self.add_nuclei_auto_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_nuclei_auto_btn.setToolTip(
            "The automatic split between background and nuclei in this box "
            "(Otsu).")
        self.add_nuclei_auto_btn.setEnabled(False)
        _nt.addWidget(self.add_nuclei_auto_btn)
        _nr.addLayout(_nt)
        self.add_nuclei_threshold_sb.editingFinished.connect(
            self._on_add_nuclei_threshold_typed)
        _mk.addWidget(self._add_nuclei_rows)
        # run rows (placed after the help: ADD_HELP_RUN, then the count --
        # mockup 2d, U-M5b)
        self.add_run_count_label = QLabel(AP.ADD_RUN_COUNT_PENDING)
        self.add_run_count_label.setWordWrap(True)
        self.add_run_count_label.setStyleSheet(_HELPER_LINE_STYLE)
        # click rows
        # Two lines: Place markers + the ball's radius, then Undo last +
        # Remove all -- one line needed 337 px in a 316 px viewport.  The
        # "7 placed" count is off the card body (built, hidden): with the
        # ball beside Place markers it does not fit, and the Markers title
        # and the first line carry the count.
        self._add_click_rows = QWidget()
        _krv = QVBoxLayout(self._add_click_rows)
        _krv.setContentsMargins(0, 0, 0, 0)
        _krv.setSpacing(2)
        _kr = QHBoxLayout()
        _kr.setContentsMargins(0, 0, 0, 0)
        _kr.setSpacing(4)
        _krv.addLayout(_kr)
        _kr2 = QHBoxLayout()
        _kr2.setContentsMargins(0, 0, 0, 0)
        _kr2.setSpacing(4)
        _krv.addLayout(_kr2)
        self.add_place_markers_btn = QPushButton("Place markers")
        self.add_place_markers_btn.setCheckable(True)
        self.add_place_markers_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_place_markers_btn.setToolTip(
            ADD_HELP_LINES["add_place_markers_btn"])
        _kr.addWidget(self.add_place_markers_btn)
        self.add_click_count_label = QLabel(AP.click_count_text(0),
                                            self._add_click_rows)
        self.add_click_count_label.setStyleSheet(PANEL_HELP_STYLE)
        self.add_click_count_label.setVisible(False)
        # the ball: "Ball" and its radius, shown with the click rows.  No
        # live typing: a value counts once it is finished, and the balls
        # are redrawn once the arrow steps stop
        self.add_click_radius_sb = QDoubleSpinBox()
        self.add_click_radius_sb.setRange(0.5, 5.0)
        self.add_click_radius_sb.setDecimals(1)
        self.add_click_radius_sb.setSingleStep(0.5)
        self.add_click_radius_sb.setValue(1.0)
        self.add_click_radius_sb.setSuffix(" µm")
        self.add_click_radius_sb.setKeyboardTracking(False)
        self.add_click_radius_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.add_click_radius_sb.setMinimumWidth(
            self._readable_spin_width(self.add_click_radius_sb))
        self._add_click_radius_row = QWidget(self._add_click_rows)
        _crr = QHBoxLayout(self._add_click_radius_row)
        _crr.setContentsMargins(0, 0, 0, 0)
        _crr.setSpacing(4)
        self._add_ball_label = QLabel(AP.ADD_BALL_LABEL,
                                      self._add_click_radius_row)
        self._add_ball_label.setStyleSheet(PANEL_ROW_LABEL_STYLE)
        _crr.addWidget(self._add_ball_label)
        # the spin and its row take the rest of the line: a row held at its
        # own hint cut the spin's " µm" and arrows off with real fonts
        _crr.addWidget(self.add_click_radius_sb, 1)
        _kr.addWidget(self._add_click_radius_row, 1)
        self.add_click_undo_btn = QPushButton("Undo last")
        self.add_click_undo_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_click_undo_btn.setToolTip(
            ADD_HELP_LINES["add_click_undo_btn"])
        _kr2.addWidget(self.add_click_undo_btn)
        # "Remove all", not "Clear": one Clear on the page (Clear objects)
        self.add_click_clear_btn = QPushButton("Remove all")
        self.add_click_clear_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_click_clear_btn.setToolTip("Remove every marker in this box.")
        _kr2.addWidget(self.add_click_clear_btn)
        # neither button keeps the keyboard: the side view clicked last
        # keeps it, so the next Ctrl+Z still reaches Add's marker undo
        # there (on the panel it would undo the active layer's paint)
        for _b in (self.add_click_undo_btn, self.add_click_clear_btn):
            _b.setFocusPolicy(Qt.NoFocus)
        _kr2.addStretch(1)
        # one checkable row per marker, "Object {id}" and
        # its z / y / x, the swatch of the id it will become.  A VIEW of
        # the orchestrator's records (the tick lives there, keyed by id);
        # hidden with no marker, on other sources and over candidates /
        # REVIEW (S5's list is the list then).  NoFocus + NoSelection: a
        # tick never takes the keyboard off the side view, so Shift+arrow
        # keeps nudging the box (M3).
        self.add_click_list = QListWidget()
        self.add_click_list.setStyleSheet(PANEL_LIST_STYLE)
        self.add_click_list.setItemDelegate(
            AddCandidateDelegate(self.add_click_list))
        self.add_click_list.setIconSize(QSize(12, 12))
        self.add_click_list.setUniformItemSizes(True)
        self.add_click_list.setSelectionMode(QListWidget.NoSelection)
        self.add_click_list.setFocusPolicy(Qt.NoFocus)
        self.add_click_list.itemChanged.connect(
            self._on_add_click_list_changed)
        # a click on a row (not on its tick) makes that object current
        self.add_click_list.itemPressed.connect(self._on_add_row_pressed)
        self.add_click_list.itemClicked.connect(self._on_add_row_clicked)
        self.add_click_list.setVisible(False)
        _krv.addWidget(self.add_click_list)
        _mk.addWidget(self._add_click_rows)
        _mk.addWidget(self._add_source_help)
        _mk.addWidget(self.add_run_count_label)
        _cc.addWidget(self._add_markers_container)

        self.add_walls_btn = self._subsection_chevron(
            "Walls", "Where each new object stops.")
        self.add_walls_btn.setProperty("_epicycle_elide_title", True)
        _cc.addWidget(self.add_walls_btn)
        self._add_walls_container = QWidget()
        _wl = QVBoxLayout(self._add_walls_container)
        _wl.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _wl.setSpacing(2)
        self.add_wall_channel_combo = QComboBox()
        self.add_wall_channel_combo.setToolTip(
            "The channel the membrane is bright in.")
        _wl.addLayout(self._refine_row("Channel", self.add_wall_channel_combo))
        self.add_wall_prep_btn = self._subsection_chevron(
            "Prepare the channel first", "Clean the channel up before the "
            "tool reads it.")
        # elided like Reshape's and Refine's prepare chips (U-H1): four
        # ticked words are wider than the viewport
        self.add_wall_prep_btn.setProperty("_epicycle_elide_title", True)
        self.add_wall_prep_btn.setStyleSheet(PANEL_CHIP_STYLE)
        _wl.addWidget(self.add_wall_prep_btn)
        self._add_wall_prep_container = QWidget()
        _wp = QVBoxLayout(self._add_wall_prep_container)
        _wp.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _wp.setSpacing(1)
        for _key in ("zcorrect", "bgsub", "lift", "denoise"):
            _cb = QCheckBox(PREP_TICK_LABELS[_key])
            _cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
            _cb.setChecked(_key == "zcorrect")
            _cb.setToolTip(_PREP_HELP.get(_key, ""))
            setattr(self, "add_prep_{}_cb".format(_key), _cb)
            _wp.addWidget(_cb)
        self._add_prep_no_clahe = QLabel(REFINE_NO_CLAHE_SENTENCE)
        self._add_prep_no_clahe.setWordWrap(True)
        self._add_prep_no_clahe.setStyleSheet(_HELPER_LINE_STYLE)
        _wp.addWidget(self._add_prep_no_clahe)
        #  take-off step (2), live-shot fix V9: the sentence moved
        # to the guide (docs/guides/annotator_3d_add.md); constructed,
        # hidden.  The sub-chevron already opens closed.
        self._add_prep_no_clahe.setVisible(False)
        self._add_wall_prep_container.setVisible(False)
        _wl.addWidget(self._add_wall_prep_container)
        self.add_wall_kind_combo = QComboBox()
        self.add_wall_kind_combo.addItems(
            [WALLS_FROM_LABELS[WALL_KINDS.index(k)] for k in ADD_WALL_KINDS])
        self.add_wall_kind_combo.setToolTip(
            "What the edge between two cells looks like in the channel.")
        _wl.addLayout(self._refine_row("Wall type", self.add_wall_kind_combo))
        # where new cells may grow, read BY INDEX through AP.OUTLINE_KEYS;
        # "Outline of the objects you cleared" is grey, the reason in its
        # own words, until a Clear is in force on this patch
        self.add_outline_combo = QComboBox()
        for _key in AP.OUTLINE_KEYS:
            self.add_outline_combo.addItem(AP.outline_item_text(_key))
        self.add_outline_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.add_outline_combo.setMinimumContentsLength(10)
        self.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        self._add_outline_row = QWidget(self._add_walls_container)
        _olr = self._refine_row(AP.ADD_OUTLINE_LABEL, self.add_outline_combo)
        _olr.setContentsMargins(0, 0, 0, 0)
        self._add_outline_row.setLayout(_olr)
        _wl.addWidget(self._add_outline_row)
        # placed, not a tooltip only: what the choice does
        self._add_outline_help = QLabel(AP.ADD_HELP_OUTLINE_NONE)
        self._add_outline_help.setWordWrap(True)
        self._add_outline_help.setStyleSheet(_HELPER_LINE_STYLE)
        _wl.addWidget(self._add_outline_help)
        self.add_show_outline_cb = QCheckBox(AP.ADD_SHOW_OUTLINE)
        self.add_show_outline_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_show_outline_cb.setVisible(False)
        _wl.addWidget(self.add_show_outline_cb)
        # under the tick while it is on: "Measuring the tissue outline…" or
        # where it is drawn
        self._add_show_outline_legend = QLabel("")
        self._add_show_outline_legend.setWordWrap(True)
        self._add_show_outline_legend.setStyleSheet(_HELPER_LINE_STYLE)
        self._add_show_outline_legend.setVisible(False)
        _wl.addWidget(self._add_show_outline_legend)
        self._add_walls_container.setVisible(False)
        _cc.addWidget(self._add_walls_container)

        # Nuclei ▸ Anywhere / the markers: the Tissue outline card, an
        # option OFF by default -- its own combo (read BY INDEX through
        # AP.OUTLINE_KEYS, Cells' items and words) and its own Show tick, so
        # each preset keeps its own choice; the same outline is measured
        # for both
        self.add_nuc_outline_btn = self._subsection_chevron(
            AP.nuclei_outline_summary(AP.OUTLINE_NONE),
            AP.ADD_HELP_NUC_OUTLINE)
        self.add_nuc_outline_btn.setProperty("_epicycle_elide_title", True)
        self.add_nuc_outline_btn.setVisible(False)
        _cc.addWidget(self.add_nuc_outline_btn)
        self._add_nuc_outline_container = QWidget()
        _no = QVBoxLayout(self._add_nuc_outline_container)
        _no.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _no.setSpacing(2)
        self.add_nuc_outline_combo = QComboBox()
        for _key in AP.OUTLINE_KEYS:
            self.add_nuc_outline_combo.addItem(AP.outline_item_text(_key))
        self.add_nuc_outline_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.add_nuc_outline_combo.setMinimumContentsLength(10)
        self.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        self._add_nuc_outline_row = QWidget(self._add_nuc_outline_container)
        _nor = self._refine_row(AP.ADD_OUTLINE_LABEL,
                                self.add_nuc_outline_combo)
        _nor.setContentsMargins(0, 0, 0, 0)
        self._add_nuc_outline_row.setLayout(_nor)
        _no.addWidget(self._add_nuc_outline_row)
        # the take-off that keeps the markers with the card open, the
        # spacing amber and an Auto status under 3.1 screens (759 -> 686 px
        # offscreen): what each choice does is its item's tooltip in the
        # list (and the guide), not a line on the page -- constructed,
        # hidden, its words kept current
        self._add_nuc_outline_help = QLabel(AP.ADD_HELP_NUC_OUTLINE_NONE)
        self._add_nuc_outline_help.setWordWrap(True)
        self._add_nuc_outline_help.setStyleSheet(_HELPER_LINE_STYLE)
        self._add_nuc_outline_help.setVisible(False)
        _no.addWidget(self._add_nuc_outline_help)
        self.add_nuc_show_outline_cb = QCheckBox(AP.ADD_SHOW_OUTLINE)
        self.add_nuc_show_outline_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_nuc_show_outline_cb.setVisible(False)
        _no.addWidget(self.add_nuc_show_outline_cb)
        self._add_nuc_show_outline_legend = QLabel("")
        self._add_nuc_show_outline_legend.setWordWrap(True)
        self._add_nuc_show_outline_legend.setStyleSheet(_HELPER_LINE_STYLE)
        self._add_nuc_show_outline_legend.setVisible(False)
        _no.addWidget(self._add_nuc_show_outline_legend)
        self._add_nuc_outline_container.setVisible(False)
        _cc.addWidget(self._add_nuc_outline_container)
        self._add_cells_container.setVisible(False)
        _s3.addWidget(self._add_cells_container)
        # ONE amber slot for what the other run or the whole patch warns
        # of (the first that applies; two notes that each fit do not fit
        # together) -- before a result only
        self._add_derive_note = QLabel("")
        self._add_derive_note.setWordWrap(True)
        self._add_derive_note.setStyleSheet(PANEL_WARN_STYLE)
        self._add_derive_note.setVisible(False)
        _s3.addWidget(self._add_derive_note)
        tune.addWidget(self._add_s3_container)

        # ==================================================================
        # S5 RESULT
        # ==================================================================
        self._add_result_container = QWidget()
        _s5 = QVBoxLayout(self._add_result_container)
        make_panel_card(self._add_result_container)
        _s5h = QHBoxLayout()
        _s5h.setContentsMargins(0, 0, 0, 0)
        _s5h.setSpacing(4)
        _s5h.addWidget(panel_card_header("Result", self._add_result_container))
        self.add_all_btn = QPushButton("All", self._add_result_container)
        self.add_all_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_all_btn.setToolTip("Tick every object in the list.")
        _s5h.addWidget(self.add_all_btn)
        self.add_none_btn = QPushButton("None", self._add_result_container)
        self.add_none_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_none_btn.setToolTip("Untick every object in the list.")
        _s5h.addWidget(self.add_none_btn)
        self.add_show_walls_cb = QCheckBox("Show what the tool sees",
                                           self._add_result_container)
        self.add_show_walls_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_show_walls_cb.setEnabled(False)
        self.add_show_walls_cb.toggled.connect(self._refresh_add_walls_legend)
        _s5h.addStretch(1)
        _s5.addLayout(_s5h)
        # U-H1: on the title row the tick set the page's minimum width, and
        # squeezed (Ignored) it clipped its own words (130 of 205 px at the
        # 316 px viewport) -- so it has a row of its own (16 px)
        _s5.addWidget(self.add_show_walls_cb)
        # the candidates as outlines, side views only.  A
        # row of its own, as the walls tick (U-H1: the title row cannot take
        # 170 px of words).
        self.add_contour_cb = QCheckBox("Show candidates as outlines",
                                        self._add_result_container)
        self.add_contour_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        _s5.addWidget(self.add_contour_cb)
        self.add_walls_legend = QLabel("", self._add_result_container)
        self.add_walls_legend.setWordWrap(True)
        self.add_walls_legend.setStyleSheet(PANEL_HELP_STYLE)
        self.add_walls_legend.setVisible(False)
        _s5.addWidget(self.add_walls_legend)
        self.add_candidate_list = QListWidget()
        self.add_candidate_list.setStyleSheet(PANEL_LIST_STYLE)
        self.add_candidate_list.setItemDelegate(
            AddCandidateDelegate(self.add_candidate_list))
        # the panel's 12 px swatch, as every other panel list: at Qt's
        # default icon size a row was 32 px, not the mockup's 19
        # (B2 fix pass U-H3, measured)
        self.add_candidate_list.setIconSize(QSize(12, 12))
        self.add_candidate_list.setUniformItemSizes(True)
        # the rows are ticked, never selected (the
        # selection was never read), and a tick must not take the focus
        # off the side view -- Shift+arrow keeps nudging the box
        self.add_candidate_list.setSelectionMode(QListWidget.NoSelection)
        self.add_candidate_list.setFocusPolicy(Qt.NoFocus)
        self.add_candidate_list.setToolTip(
            "The new objects. Tick the ones to add; Accept adds exactly what "
            "is ticked.")
        self.add_candidate_list.itemChanged.connect(
            self._on_add_candidate_check_changed)
        self.add_candidate_list.itemPressed.connect(self._on_add_row_pressed)
        self.add_candidate_list.itemClicked.connect(self._on_add_row_clicked)
        _s5.addWidget(self.add_candidate_list)
        # The counts note: at most ADD_NOTES_LINES lines, ELIDED (not
        # clipped), top-aligned, the full text in its tooltip
        # (B2 fix pass U-M2 -- a height cap on a vertically centred label
        # cut the text at the top AND the bottom).  Horizontally Ignored
        # so its unwrapped lines never set the page's width.
        self.add_notes_label = QLabel("", self._add_result_container)
        self.add_notes_label.setWordWrap(False)
        self.add_notes_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.add_notes_label.setStyleSheet(PANEL_HELP_STYLE)
        self.add_notes_label.setSizePolicy(
            QSizePolicy.Ignored,
            self.add_notes_label.sizePolicy().verticalPolicy())
        self.add_notes_label.setVisible(False)
        self._add_notes_elider = _ElideOnResize(
            self._apply_add_notes_elision, self.add_notes_label)
        self.add_notes_label.installEventFilter(self._add_notes_elider)
        _s5.addWidget(self.add_notes_label)
        self.add_all_btn.clicked.connect(
            lambda _c=False: self._set_all_add_checks(True))
        self.add_none_btn.clicked.connect(
            lambda _c=False: self._set_all_add_checks(False))
        self._add_result_container.setVisible(False)
        tune.addWidget(self._add_result_container)

        # ==================================================================
        # ADVANCED
        # ==================================================================
        self._add_advanced_toggle = self._subsection_chevron(
            "Advanced", "Knobs you rarely need.")
        self._add_advanced_toggle.setProperty("_epicycle_elide_title", True)
        tune.addWidget(self._add_advanced_toggle)
        self._add_advanced_container = QWidget()
        _adv = QVBoxLayout(self._add_advanced_container)
        _adv.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _adv.setSpacing(2)

        def _dspin(lo, hi, dec, val, step, suffix=""):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
            sb.setSingleStep(step)
            sb.setValue(val)
            if suffix:
                sb.setSuffix(suffix)
            return sb

        self._add_cells_adv = QWidget()
        _ca = QVBoxLayout(self._add_cells_adv)
        _ca.setContentsMargins(0, 0, 0, 0)
        _ca.setSpacing(2)
        # a Cells whole patch: what fills it under the Grow by pick, at the
        # top of the card
        self._add_whole_fill_label = QLabel(AP.ADD_HELP_WHOLE_FILL,
                                            self._add_cells_adv)
        self._add_whole_fill_label.setWordWrap(True)
        self._add_whole_fill_label.setStyleSheet(_HELPER_LINE_STYLE)
        self._add_whole_fill_label.setVisible(False)
        _ca.addWidget(self._add_whole_fill_label)
        # Grow by, read BY INDEX through AP.GROW_KEYS: the fill as before,
        # or Spread from each marker (on a box and on a whole patch)
        self.add_grow_combo = QComboBox()
        self.add_grow_combo.addItems([AP.ADD_GROW_FLOOD, AP.ADD_GROW_SPREAD])
        self.add_grow_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.add_grow_combo.setMinimumContentsLength(10)
        self._add_grow_row = QWidget(self._add_cells_adv)
        _grr = self._refine_row(AP.ADD_GROW_LABEL, self.add_grow_combo,
                                advanced=True)
        _grr.setContentsMargins(0, 0, 0, 0)
        self._add_grow_row.setLayout(_grr)
        _ca.addWidget(self._add_grow_row)
        # Spread's own two rows (one wrapper each), shown with Spread
        self._add_spread_adv = QWidget(self._add_cells_adv)
        _sp = QVBoxLayout(self._add_spread_adv)
        _sp.setContentsMargins(0, 0, 0, 0)
        _sp.setSpacing(2)
        # the engine's ranges (0-100, 2-40, 0-30: ``test_add_outline_wiring``
        # pins them to the engine's own) and its defaults
        from packages.epicycle_annotator.reshape import spec as _RS

        self.add_wall_cost_sb = _dspin(0.0,
                                       100.0, 0,
                                       float(_RS.DEFAULT_PARAMS["wall_cost"]),
                                       5.0)
        self._add_wall_cost_row = QWidget(self._add_spread_adv)
        _wcr = self._refine_row(AP.ADD_WALL_COST_LABEL, self.add_wall_cost_sb,
                                advanced=True)
        _wcr.setContentsMargins(0, 0, 0, 0)
        self._add_wall_cost_row.setLayout(_wcr)
        _sp.addWidget(self._add_wall_cost_row)
        self.add_growth_limit_sb = _dspin(
            2.0, 40.0, 0,
            float(_RS.DEFAULT_PARAMS["growth_limit"]), 1.0)
        self._add_growth_limit_row = QWidget(self._add_spread_adv)
        _glr = self._refine_row(AP.ADD_GROWTH_LIMIT_LABEL,
                                self.add_growth_limit_sb, advanced=True)
        _glr.setContentsMargins(0, 0, 0, 0)
        self._add_growth_limit_row.setLayout(_glr)
        _sp.addWidget(self._add_growth_limit_row)
        self._add_spread_adv.setVisible(False)
        _ca.addWidget(self._add_spread_adv)
        # Include membrane: with Spread on a box, and on a whole patch (the
        # fill hands the membrane at a new cell's edge to it too)
        self.add_include_membrane_cb = QCheckBox(AP.ADD_INCLUDE_MEMBRANE,
                                                 self._add_cells_adv)
        self.add_include_membrane_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_include_membrane_cb.setChecked(
            bool(_RS.DEFAULT_PARAMS["include_membrane"]))
        self.add_include_membrane_cb.setVisible(False)
        _ca.addWidget(self.add_include_membrane_cb)
        # the run source's own row, shown only with that source (hide,
        # don't grey): the shrink with another run's objects (the click
        # ball's radius sits beside Place markers)
        self.add_core_erode_sb = _dspin(0.0, 5.0, 1, 1.0, 0.5, " µm")
        self._add_core_erode_row = QWidget(self._add_cells_adv)
        _cer = self._refine_row("Shrink run objects", self.add_core_erode_sb,
                                advanced=True)
        _cer.setContentsMargins(0, 0, 0, 0)
        self._add_core_erode_row.setLayout(_cer)
        _ca.addWidget(self._add_core_erode_row)
        self.add_dark_outside_cb = QCheckBox(AP.ADD_DARK_OUTSIDE_LABEL)
        self.add_dark_outside_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_dark_outside_cb.setChecked(True)
        self.add_dark_outside_cb.setToolTip(AP.ADD_HELP_DARK_OUTSIDE)
        _ca.addWidget(self.add_dark_outside_cb)
        self.add_dark_clearance_sb = _dspin(0.0, 10.0, 1, 2.0, 0.5, " µm")
        self.add_dark_clearance_sb.setToolTip(AP.ADD_HELP_DARK)
        # Dark space reach and Dark below as row widgets (no margins: the
        # same height as the bare rows), so Nuclei's shrink-wrap can show
        # Dark below alone
        self._add_dark_reach_row = QWidget(self._add_cells_adv)
        _drr = self._refine_row("Dark space reach",
                                self.add_dark_clearance_sb, advanced=True)
        _drr.setContentsMargins(0, 0, 0, 0)
        self._add_dark_reach_row.setLayout(_drr)
        _ca.addWidget(self._add_dark_reach_row)
        self.add_dark_factor_sb = _dspin(0.05, 1.0, 2, 0.5, 0.05)
        self.add_dark_factor_sb.setToolTip(AP.ADD_HELP_DARK_BELOW)
        # the shrink-wrap is measured again only when a value is FINISHED,
        # not on every key press
        self.add_dark_factor_sb.setKeyboardTracking(False)
        self._add_dark_below_row = QWidget(self._add_cells_adv)
        _dbr = self._refine_row("Dark below", self.add_dark_factor_sb,
                                advanced=True)
        _dbr.setContentsMargins(0, 0, 0, 0)
        self._add_dark_below_row.setLayout(_dbr)
        _ca.addWidget(self._add_dark_below_row)
        # the shrink-wrap's closing (Shrink-wrap around the stain only)
        self.add_bridge_sb = _dspin(0.0,
                                    30.0, 1,
                                    AP.ADD_BRIDGE_UM, 1.0, " µm")
        self.add_bridge_sb.setKeyboardTracking(False)
        self._add_bridge_row = QWidget(self._add_cells_adv)
        _brr = self._refine_row(AP.ADD_BRIDGE_LABEL, self.add_bridge_sb,
                                advanced=True)
        _brr.setContentsMargins(0, 0, 0, 0)
        self._add_bridge_row.setLayout(_brr)
        self._add_bridge_row.setVisible(False)
        _ca.addWidget(self._add_bridge_row)
        # on screen, not only a tooltip (mockup 8; U-M6).  Advanced is
        # outside the ceiling; its other knobs stay tooltip-only (a
        # recorded deviation from the mockup, which draws only this one).
        self._add_dark_help = QLabel(AP.ADD_HELP_DARK)
        self._add_dark_help.setWordWrap(True)
        self._add_dark_help.setStyleSheet(_HELPER_LINE_STYLE)
        _ca.addWidget(self._add_dark_help)
        _adv.addWidget(self._add_cells_adv)

        self._add_nuclei_adv = QWidget()
        _nadv = QVBoxLayout(self._add_nuclei_adv)
        _nadv.setContentsMargins(0, 0, 0, 0)
        _nadv.setSpacing(2)
        # Anywhere's three rows and Inside's three rows: one set shows
        self._add_anywhere_adv = QWidget(self._add_nuclei_adv)
        _na = QVBoxLayout(self._add_anywhere_adv)
        _na.setContentsMargins(0, 0, 0, 0)
        _na.setSpacing(2)
        _nadv.addWidget(self._add_anywhere_adv)
        self._add_inside_adv = QWidget(self._add_nuclei_adv)
        _ia = QVBoxLayout(self._add_inside_adv)
        _ia.setContentsMargins(0, 0, 0, 0)
        _ia.setSpacing(2)
        from packages.epicycle_annotator import add_derive as _AD

        self.add_nuc_radius_sb = _dspin(_AD.NUC_RADIUS_RANGE[0],
                                        _AD.NUC_RADIUS_RANGE[1], 1,
                                        _AD.NUC_RADIUS_UM, 0.5, " µm")
        _ia.addLayout(self._refine_row("Nucleus radius",
                                       self.add_nuc_radius_sb, advanced=True))
        self.add_nuc_edge_sb = _dspin(_AD.NUC_EDGE_RANGE[0],
                                      _AD.NUC_EDGE_RANGE[1], 2,
                                      _AD.NUC_EDGE, 0.05)
        _ia.addLayout(self._refine_row("Nucleus edge", self.add_nuc_edge_sb,
                                       advanced=True))
        self.add_two_gap_sb = _dspin(_AD.TWO_GAP_RANGE[0],
                                     _AD.TWO_GAP_RANGE[1], 2,
                                     _AD.TWO_GAP, 0.05)
        _ia.addLayout(self._refine_row("Gap that splits two nuclei",
                                       self.add_two_gap_sb, advanced=True))
        self._add_inside_adv.setVisible(False)
        _nadv.addWidget(self._add_inside_adv)
        self.add_separate_cb = QCheckBox("Separate touching objects")
        self.add_separate_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_separate_cb.setChecked(True)
        self.add_separate_cb.setToolTip(
            "Split objects that touch into separate ones. Off: one object "
            "per bright blob.")
        _na.addWidget(self.add_separate_cb)
        self.add_min_distance_sb = QSpinBox()
        self.add_min_distance_sb.setRange(1, 30)
        self.add_min_distance_sb.setValue(3)
        self.add_min_distance_sb.setToolTip(
            "The closest two object centres may be, in voxels. Lower gives "
            "more, smaller objects.")
        # a row widget (no margins: the same height as the bare row), so
        # the markers can show their own Spacing in its place
        self._add_spacing_row = QWidget(self._add_anywhere_adv)
        _spr = self._refine_row("Spacing", self.add_min_distance_sb,
                                advanced=True)
        _spr.setContentsMargins(0, 0, 0, 0)
        self._add_spacing_row.setLayout(_spr)
        _na.addWidget(self._add_spacing_row)
        # the markers' own Spacing, a twin (a preset application writes
        # Anywhere's untouched box back to 3, and what the user touched is
        # kept per box), in its own row of the same height, shown in place
        # of Anywhere's on the markers only
        self.add_marker_spacing_sb = QSpinBox()
        self.add_marker_spacing_sb.setRange(1, 30)
        self.add_marker_spacing_sb.setValue(_AP_ADD_MARKER_SPACING_DEFAULT)
        self._add_marker_spacing_row = QWidget(self._add_anywhere_adv)
        _mspr = self._refine_row("Spacing", self.add_marker_spacing_sb,
                                 advanced=True)
        _mspr.setContentsMargins(0, 0, 0, 0)
        self._add_marker_spacing_row.setLayout(_mspr)
        self._add_marker_spacing_row.setVisible(False)
        _na.addWidget(self._add_marker_spacing_row)
        self.add_clahe_clip_sb = _dspin(0.001, 1.0, 3, 0.01, 0.01)
        self.add_clahe_clip_sb.setToolTip(
            "How strongly Local contrast evens the box out.")
        _na.addLayout(self._refine_row("Local contrast strength",
                                       self.add_clahe_clip_sb, advanced=True))
        _adv.addWidget(self._add_nuclei_adv)

        self.add_clear_border_cb = QCheckBox(
            "Untick objects at the box edge")
        self.add_clear_border_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_clear_border_cb.setChecked(True)
        self.add_clear_border_cb.setToolTip(
            "Objects that reach the box edge were probably cut by it; they "
            "start unticked. A lone object stays ticked.")
        _adv.addWidget(self.add_clear_border_cb)
        self.add_edge_margin_sb = QSpinBox()
        self.add_edge_margin_sb.setRange(0, 20)
        self.add_edge_margin_sb.setValue(3)
        self.add_edge_margin_sb.setToolTip(
            "How many voxels in from the box edge count as the edge.")
        # hidden with the tick on a whole patch (no face is a box edge)
        self._add_edge_zone_row = QWidget(self._add_advanced_container)
        _ezr = self._refine_row("Edge zone", self.add_edge_margin_sb,
                                advanced=True)
        _ezr.setContentsMargins(0, 0, 0, 0)
        self._add_edge_zone_row.setLayout(_ezr)
        _adv.addWidget(self._add_edge_zone_row)
        self.add_min_size_sb = QSpinBox()
        self.add_min_size_sb.setRange(0, 1000000)
        self.add_min_size_sb.setValue(0)
        self.add_min_size_sb.setToolTip(
            "Objects smaller than this many voxels are left out. 0 keeps "
            "every one.")
        _adv.addLayout(self._refine_row("Smallest object",
                                        self.add_min_size_sb, advanced=True))
        self.add_auto_smooth_cb = QCheckBox("Smooth the added objects")
        self.add_auto_smooth_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_auto_smooth_cb.setChecked(True)
        self.add_auto_smooth_cb.setToolTip(
            "Smooths each new object's surface in the preview, so what you "
            "see is what Accept writes.")
        _adv.addWidget(self.add_auto_smooth_cb)
        self.add_smooth_sigma_sb = _dspin(0.1, 10.0, 1, 1.5, 0.5)
        self.add_smooth_sigma_sb.setToolTip(
            "How much smoothing, in voxels of the full-size image. 1.5 "
            "is gentle.")
        _adv.addLayout(self._refine_row("Smoothing", self.add_smooth_sigma_sb,
                                        advanced=True))
        # the markers run after every change while this is on (shown with
        # the click markers only; kept per workspace by the orchestrator)
        self.add_live_cb = QCheckBox(AP.ADD_LIVE_LABEL)
        self.add_live_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_live_cb.setChecked(True)
        self.add_live_cb.setToolTip(AP.ADD_HELP_LIVE_TICK)
        self.add_live_cb.setVisible(False)
        _adv.addWidget(self.add_live_cb)
        # directly under it, grey: why a Cells click is not run as you
        # click here (the orchestrator's sentence, decided with S1's);
        # hidden while live runs, with the tick off, and while "Always
        # update as I click" is on
        self._add_live_note = QLabel("")
        self._add_live_note.setWordWrap(True)
        self._add_live_note.setStyleSheet(_HELPER_LINE_STYLE)
        self._add_live_note.setVisible(False)
        _adv.addWidget(self._add_live_note)
        # the live restrictions switched off (Cells ▸ Click markers with the
        # tick above on only; off by default and never kept: the
        # orchestrator does not save it)
        self.add_live_always_cb = QCheckBox(AP.ADD_LIVE_ALWAYS_LABEL)
        self.add_live_always_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.add_live_always_cb.setChecked(False)
        self.add_live_always_cb.setToolTip(AP.ADD_HELP_LIVE_ALWAYS)
        self.add_live_always_cb.setVisible(False)
        _adv.addWidget(self.add_live_always_cb)
        self._add_advanced_help = QLabel("")
        self._add_advanced_help.setWordWrap(True)
        self._add_advanced_help.setStyleSheet(PANEL_WARN_STYLE)
        self._add_advanced_help.setVisible(False)
        _adv.addWidget(self._add_advanced_help)
        self._add_advanced_container.setVisible(False)
        tune.addWidget(self._add_advanced_container)
        self._add_advanced_toggle.toggled.connect(self._on_add_advanced_toggled)

        layout.addWidget(self._add_tune_container)

        # ==================================================================
        # REVIEW (replaces S2 / S3 / S5 / Advanced in place)
        # ==================================================================
        self._add_review_container = QWidget()
        review = QVBoxLayout(self._add_review_container)
        make_panel_card(self._add_review_container)
        review.addWidget(panel_card_header("Added this session",
                                           self._add_review_container))
        self.add_pieces_list = QListWidget()
        self.add_pieces_list.setStyleSheet(PANEL_LIST_STYLE)
        self.add_pieces_list.setItemDelegate(
            AddCandidateDelegate(self.add_pieces_list))
        self.add_pieces_list.setIconSize(QSize(12, 12))
        self.add_pieces_list.setUniformItemSizes(True)
        self.add_pieces_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.add_pieces_list.setToolTip(
            "Every object added this session. Select one to refine it.")
        review.addWidget(self.add_pieces_list)
        # ONE row: Refine all ▸ · Refine ▸ · + Add another (Undo all is the
        # footer's secondary in REVIEW, beside Done)
        self._add_review_btn_row = QWidget(self._add_review_container)
        piece_btn_row = QHBoxLayout(self._add_review_btn_row)
        piece_btn_row.setContentsMargins(0, 0, 0, 0)
        # 5 px side padding and 3 px gaps: the three fit the 312 px live width
        piece_btn_row.setSpacing(3)
        self.add_batch_refine_btn = QPushButton(ADD_REFINE_ALL_BTN)
        self.add_batch_refine_btn.setStyleSheet(_ADD_TIGHT_BTN_STYLE)
        self.add_batch_refine_btn.setToolTip(ADD_REFINE_ALL_TIP)
        self.add_batch_refine_btn.setVisible(False)
        piece_btn_row.addWidget(self.add_batch_refine_btn)
        self.add_refine_piece_btn = QPushButton("Refine ▸")
        self.add_refine_piece_btn.setStyleSheet(_ADD_TIGHT_BTN_STYLE)
        self.add_refine_piece_btn.setEnabled(False)
        self.add_refine_piece_btn.setToolTip(
            "Take the object selected in the list to the Refine tab.")
        piece_btn_row.addWidget(self.add_refine_piece_btn)
        self.add_merge_pieces_btn = QPushButton("Merge selected")
        self.add_merge_pieces_btn.setEnabled(False)
        self.add_merge_pieces_btn.setToolTip(
            "Merge two over-split objects back into one (coming in a later "
            "version).")
        self.add_merge_pieces_btn.setVisible(False)
        piece_btn_row.addWidget(self.add_merge_pieces_btn)
        self.add_again_btn = QPushButton("+ Add another")
        self.add_again_btn.setStyleSheet(_ADD_TIGHT_BTN_STYLE)
        self.add_again_btn.setToolTip(AP.ADD_AGAIN_TIP)
        piece_btn_row.addWidget(self.add_again_btn)
        review.addWidget(self._add_review_btn_row)
        self._add_review_help = QLabel("")
        self._add_review_help.setWordWrap(True)
        self._add_review_help.setStyleSheet(_HELPER_LINE_STYLE)
        review.addWidget(self._add_review_help)
        self._add_review_container.setVisible(False)
        layout.addWidget(self._add_review_container)

        layout.addStretch(1)

        # ==================================================================
        # S4 PINNED -- Preview, Stop, the busy bar, the stage line
        # ==================================================================
        self._add_s4_container = QWidget(page)
        _s4 = QVBoxLayout(self._add_s4_container)
        _s4.setContentsMargins(0, 0, 0, 0)
        _s4.setSpacing(2)
        _prow = QWidget()
        _prl = QHBoxLayout(_prow)
        _prl.setContentsMargins(0, 0, 0, 0)
        _prl.setSpacing(4)
        # Clear objects: ONE menu button in the pinned row (0 px of height
        # in every state), Qt's own arrow on the right (the Refine Recipes
        # rule, no arrow character in the text).  Shown while a box is
        # placed, no candidates are up and not in the review.
        self.add_clear_menu = QMenu(self)
        self.add_clear_menu.setToolTipsVisible(True)
        self.add_clear_box_action = self.add_clear_menu.addAction(
            AP.ADD_CLEAR_ITEM_BOX)
        self.add_clear_patch_action = self.add_clear_menu.addAction(
            AP.ADD_CLEAR_ITEM_PATCH)
        self.add_clear_box_action.triggered.connect(self._fire_add_clear_box)
        self.add_clear_patch_action.triggered.connect(
            self._fire_add_clear_patch)
        self.add_clear_btn = QPushButton(AP.ADD_CLEAR_BTN)
        self.add_clear_btn.setStyleSheet(
            PANEL_BTN_STYLE + _RECIPES_MENU_INDICATOR_STYLE)
        self.add_clear_btn.setMenu(self.add_clear_menu)
        self.add_clear_btn.setVisible(False)
        _prl.addWidget(self.add_clear_btn)
        self.add_preview_btn = QPushButton("Preview")
        self.add_preview_btn.setStyleSheet(PANEL_PRIMARY_BTN_STYLE)
        self.add_preview_btn.setEnabled(False)
        self.add_preview_btn.setToolTip(
            "Find the new objects in the box. Nothing is written until "
            "Accept.")
        _prl.addWidget(self.add_preview_btn, 1)
        self.add_stop_btn = QPushButton("Stop")
        self.add_stop_btn.setStyleSheet(PANEL_BTN_STYLE)
        self.add_stop_btn.setToolTip(
            "Stop the preview. Nothing from this box is added.")
        self.add_stop_btn.setEnabled(False)
        _prl.addWidget(self.add_stop_btn)
        _s4.addWidget(_prow)
        self._add_busy_container = QWidget()
        _bz = QHBoxLayout(self._add_busy_container)
        _bz.setContentsMargins(0, 0, 0, 0)
        self.add_busy_bar = QProgressBar()
        self.add_busy_bar.setRange(0, 0)
        self.add_busy_bar.setTextVisible(False)
        self.add_busy_bar.setFixedHeight(4)
        self.add_busy_bar.setStyleSheet(_PANEL_BUSY_BAR_STYLE)
        _bz.addWidget(self.add_busy_bar, 1)
        _s4.addWidget(self._add_busy_container)
        self.add_stage_label = QLabel("X")
        self.add_stage_label.setStyleSheet(_HELPER_LINE_STYLE)
        self.add_stage_label.setWordWrap(False)
        self.add_stage_label.setSizePolicy(
            QSizePolicy.Ignored,
            self.add_stage_label.sizePolicy().verticalPolicy())
        self._add_stage_full_text = ""
        self._add_stage_elider = _ElideOnResize(
            self._apply_add_stage_elision, self.add_stage_label)
        self.add_stage_label.installEventFilter(self._add_stage_elider)
        _s4.addWidget(self.add_stage_label)
        _reserved = max(
            self._add_s4_container.sizeHint().height(),
            _prow.sizeHint().height() + 2 * _s4.spacing()
            + self.add_busy_bar.height()
            + self.add_stage_label.sizeHint().height(),
        )
        # The mockup draws Stop, the bar and the stage line in EVERY state,
        # dimmed when idle (live-shot fix V2); the strip's height is fixed,
        # so showing them costs nothing below it.
        self._set_add_strip_busy(False)
        self.add_stage_label.setText("")
        self._add_s4_container.setFixedHeight(int(_reserved))
        page.layout().insertWidget(page.layout().indexOf(footer),
                                   self._add_s4_container, 0)

        # ==================================================================
        # S6 FOOTER
        # ==================================================================
        _ft = QVBoxLayout(footer)
        _ft.setContentsMargins(4, 2, 4, 2)
        _ft.setSpacing(2)
        add_btn_row = QHBoxLayout()
        self.add_accept_btn = QPushButton("Accept")
        self.add_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.add_accept_btn.setEnabled(False)
        # the secondary: one word per meaning -- Close box (a box is up, and
        # grey with the same word before one is), Discard (a result is on
        # screen), Undo all (REVIEW); grey while busy (Stop is the one exit)
        self.add_reject_btn = QPushButton("Close box")
        self.add_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)
        self.add_reject_btn.setEnabled(False)
        self._add_reject_wanted = False
        add_btn_row.addWidget(self.add_accept_btn)
        add_btn_row.addWidget(self.add_reject_btn)
        _ft.addLayout(add_btn_row)
        self._add_footer_summary = self._footer_summary_label()
        self._add_footer_slot = QWidget(footer)
        _slot = QVBoxLayout(self._add_footer_slot)
        _slot.setContentsMargins(0, 0, 0, 0)
        _slot.setSpacing(0)
        _slot.addWidget(self._add_footer_summary)
        self._add_footer_slot.setFixedHeight(
            self._add_footer_summary.height()
            if self._add_footer_summary.height() > 0
            else self._add_footer_summary.maximumHeight())
        _ft.addWidget(self._add_footer_slot)
        self._refresh_add_footer_summary()

        # -- mutual exclusion + the touched set + the preset --------------
        self.add_markers_btn.toggled.connect(
            lambda c: self._on_add_subsection_toggled("add_markers_btn", c))
        self.add_walls_btn.toggled.connect(
            lambda c: self._on_add_subsection_toggled("add_walls_btn", c))
        self.add_nuc_outline_btn.toggled.connect(
            _partial(self._on_add_subsection_toggled, "add_nuc_outline_btn"))
        self.add_wall_prep_btn.toggled.connect(self._on_add_wall_prep_toggled)
        for _attr in _ADD_PRESET_KEYS:
            _w = getattr(self, _attr, None)
            if _w is None:
                continue
            if isinstance(_w, QCheckBox):
                _sig = _w.toggled
            elif isinstance(_w, QComboBox):
                _sig = _w.currentIndexChanged
            else:
                _sig = _w.valueChanged
            _sig.connect(lambda *_a, a=_attr: self._mark_add_touched(a))
        self.add_preset_combo.currentIndexChanged.connect(
            self._on_add_preset_index_changed)
        self.add_marker_source_combo.currentIndexChanged.connect(
            self._on_add_source_index_changed)
        self.add_look_for_combo.currentIndexChanged.connect(
            self._on_add_look_for_index_changed)
        for _w in (self.add_nuc_radius_sb, self.add_nuc_edge_sb,
                   self.add_two_gap_sb):
            _w.valueChanged.connect(lambda *_a: self._refresh_add_summaries())
        self.add_whole_btn.clicked.connect(self._fire_add_whole)
        self.add_size_whole_btn.clicked.connect(self._fire_add_whole)
        self.add_dark_outside_cb.toggled.connect(
            lambda _c: self._apply_add_visibility())
        # the markers' Spacing shows only while the tick is on
        self.add_separate_cb.toggled.connect(self._on_add_separate_toggled)
        # Grow by, Tissue outline and Bridge gaps: the user's choice is kept
        # (a default never overwrites it), and each redraws the page
        for _attr, _sig in (("add_grow_combo", "currentIndexChanged"),
                            ("add_outline_combo", "currentIndexChanged"),
                            ("add_bridge_sb", "valueChanged")):
            getattr(getattr(self, _attr), _sig).connect(
                lambda *_a, a=_attr: self._mark_add_touched(a))
        self.add_grow_combo.currentIndexChanged.connect(
            lambda *_a: (self._apply_add_visibility(),
                         self._refresh_add_summaries()))
        # "Always update as I click" shows only while the live tick is on
        self.add_live_cb.toggled.connect(
            lambda _c: self._apply_add_visibility())
        # the line under the live tick hides while this is on
        self.add_live_always_cb.toggled.connect(
            lambda _c: self._apply_add_visibility())
        self.add_outline_combo.currentIndexChanged.connect(
            self._on_add_outline_index_changed)
        self.add_show_outline_cb.toggled.connect(
            lambda _c: self._apply_add_visibility())
        # Nuclei's own Tissue outline: the user's pick is kept (its untouched
        # default is always No outline), its own slot, its own Show tick
        self.add_nuc_outline_combo.currentIndexChanged.connect(
            lambda *_a: self._mark_add_touched("add_nuc_outline_combo"))
        self.add_nuc_outline_combo.currentIndexChanged.connect(
            self._on_add_nuc_outline_index_changed)
        self.add_nuc_show_outline_cb.toggled.connect(
            lambda _c: self._apply_add_visibility())
        for _w in (self.add_wall_cost_sb, self.add_growth_limit_sb,
                   self.add_include_membrane_cb, self.add_bridge_sb):
            _sig = (_w.toggled if isinstance(_w, QCheckBox)
                    else _w.valueChanged)
            _sig.connect(lambda *_a: self._refresh_add_summaries())
        for _w in (self.add_click_radius_sb, self.add_core_erode_sb,
                   self.add_dark_clearance_sb, self.add_dark_factor_sb,
                   self.add_separate_cb, self.add_min_distance_sb,
                   self.add_marker_spacing_sb,
                   self.add_clahe_clip_sb, self.add_edge_margin_sb,
                   self.add_min_size_sb, self.add_smooth_sigma_sb,
                   self.add_auto_smooth_cb, self.add_clear_border_cb,
                   self.add_dark_outside_cb):
            _sig = (_w.toggled if isinstance(_w, QCheckBox)
                    else _w.valueChanged)
            _sig.connect(lambda *_a: self._refresh_add_summaries())
        for _key in ("zcorrect", "bgsub", "lift", "denoise"):
            getattr(self, "add_prep_{}_cb".format(_key)).toggled.connect(
                lambda *_a: self._refresh_add_summaries())
        self.add_wall_kind_combo.currentIndexChanged.connect(
            lambda *_a: self._refresh_add_summaries())
        for _combo in (self.add_nuclei_channel_combo,
                       self.add_wall_channel_combo):
            _combo.currentIndexChanged.connect(
                lambda *_a: self._refresh_add_summaries())
        # the Auto help slot reads the nuclei channel (live-shot fix F9)
        self.add_nuclei_channel_combo.currentIndexChanged.connect(
            lambda *_a: self._apply_add_visibility())
        self.set_add_marker_sources([])
        # Look for nuclei offers the markers before any run is listed
        self.set_add_look_for_items([])
        self._set_add_channel_blank(self.add_nuclei_channel_combo)
        self._set_add_channel_blank(self.add_wall_channel_combo)
        self._apply_add_visibility()
        self._refresh_add_summaries()
        for _attr, _tip in ADD_HELP_LINES.items():
            _w = getattr(self, _attr, None)
            if _w is not None:
                _w.setToolTip(_tip)
        #  (M3), measured: a press on a NoFocus list hands the
        # focus to the first ancestor that takes click focus -- the page's
        # scroll area -- so ticking a marker row took the keyboard off the
        # side view and Shift+arrow scrolled the page instead of nudging
        # the box.  The Add page's scroll area takes no focus, so the walk
        # reaches the window and the side view keeps it.  (Every input on
        # the page still takes its own focus on click.)
        scroll.setFocusPolicy(Qt.NoFocus)

        self._register_page(
            "Add", page, scroll, content, footer,
            status=self.add_status_label, preview=self.add_preview_btn,
        )
        return page

    # --- Add page helpers ---------------------------

    def _open_add_numbers(self) -> None:
        dlg = getattr(self, "_add_numbers_dialog", None)
        if dlg is None:
            return
        cb = getattr(self, "_on_add_numbers_cb", None)
        if cb is not None:
            try:
                if cb() is False:
                    return
            except Exception:
                pass
        dlg.show()
        dlg.raise_()

    def close_add_numbers(self) -> None:
        """Closed at every Add exit."""
        dlg = getattr(self, "_add_numbers_dialog", None)
        if dlg is not None:
            try:
                dlg.hide()
            except Exception:
                pass

    def _apply_add_stage_elision(self) -> None:
        lb = getattr(self, "add_stage_label", None)
        if lb is None:
            return
        full = str(getattr(self, "_add_stage_full_text", "") or "")
        width = lb.contentsRect().width()
        if not lb.isVisible() or width <= 0:
            lb.setText(full)
            return
        lb.setText(lb.fontMetrics().elidedText(full, Qt.ElideRight, width))

    def _on_add_advanced_toggled(self, checked: bool) -> None:
        self._add_advanced_container.setVisible(
            bool(checked) and not self._add_advanced_toggle.isHidden())
        self._refresh_chevron_text(self._add_advanced_toggle)

    def _on_add_wall_prep_toggled(self, checked: bool) -> None:
        self._add_wall_prep_container.setVisible(bool(checked))
        self._refresh_chevron_text(self.add_wall_prep_btn)

    #: the page's cards (chevron, body): one open at a time among the
    #: chevrons on the page
    _ADD_CARDS = (("add_markers_btn", "_add_markers_container"),
                  ("add_walls_btn", "_add_walls_container"),
                  ("add_nuc_outline_btn", "_add_nuc_outline_container"))

    def _on_add_subsection_toggled(self, attr: str, checked: bool) -> None:
        """Markers, Walls and (Nuclei) Tissue outline: opening one closes
        the others.  Only this preset's cards take part: a card that is not
        on this preset (Walls on Nuclei, Nuclei's Tissue outline on Cells)
        is left as it is, so Cells comes back with its cards as the user
        left them.  Nuclei's Tissue outline takes part on Nuclei even while
        it is hidden for a moment (candidates up, a whole patch), so it never
        comes back open beside the card opened since.  The bodies are
        decided in ``_apply_add_visibility``."""
        if getattr(self, "_add_subsection_syncing", False):
            return
        self._add_subsection_syncing = True
        try:
            if checked:
                nuclei = self.get_add_preset() != "cells"
                for other, _body in self._ADD_CARDS:
                    if other == attr:
                        continue
                    btn = getattr(self, other, None)
                    if (btn is not None and btn.isChecked()
                            and (self._add_chevron_shown(other)
                                 or (nuclei
                                     and other == "add_nuc_outline_btn"))):
                        btn.setChecked(False)
        finally:
            self._add_subsection_syncing = False
        self._apply_add_visibility()
        for btn_attr, _body in self._ADD_CARDS:
            btn = getattr(self, btn_attr, None)
            if btn is not None:
                self._refresh_chevron_text(btn)

    def _apply_add_card_bodies(self) -> None:
        """A card's body shows only when its chevron is on the page AND
        open: a card left open on Cells never shows its body on Nuclei.  On
        Nuclei the two cards are never open together: Markers wins."""
        if (hasattr(self, "add_nuc_outline_btn")
                and self.get_add_preset() != "cells"
                and self.add_nuc_outline_btn.isChecked()
                and self.add_markers_btn.isChecked()
                and self._add_chevron_shown("add_markers_btn")
                and self._add_chevron_shown("add_nuc_outline_btn")):
            btn = self.add_nuc_outline_btn
            btn.blockSignals(True)
            try:
                btn.setChecked(False)
            finally:
                btn.blockSignals(False)
            self._refresh_chevron_text(btn)
        for btn_attr, body_attr in self._ADD_CARDS:
            if not hasattr(self, btn_attr):
                continue
            shown = self._add_chevron_shown(btn_attr)
            getattr(self, btn_attr).setVisible(shown)
            getattr(self, body_attr).setVisible(
                shown and getattr(self, btn_attr).isChecked())

    @staticmethod
    def _set_add_channel_blank(combo) -> None:
        try:
            combo.setCurrentIndex(-1)
        except Exception:
            pass

    def _on_add_separate_toggled(self, _checked=False) -> None:
        """Separate touching objects changed: the markers' Spacing row
        follows it (shown only while it is ticked)."""
        self._apply_add_visibility()

    def _mark_add_touched(self, attr: str) -> None:
        if not getattr(self, "_add_applying_preset", False):
            self._add_touched.add(attr)

    def clear_add_touched(self, *, keep_source: bool = False) -> None:
        """A full Add exit forgets which keys the user touched (4.1a).

        ``keep_source`` keeps the Markers source the user picked: the end
        of a box or a whole patch passes it, so the next one starts from
        that source (a run picked under Markers is not put back to the
        preset's default).  Another patch or run forgets it
        (``forget_add_source``)."""
        keep = "add_marker_source_combo"
        kept = keep_source and keep in self._add_touched
        self._add_touched = {keep} if kept else set()

    def forget_add_source(self) -> None:
        """Markers back to the preset's default source, and no longer the
        user's pick (another patch or run: the run picked belongs to the
        patch it was picked on).  A pick set aside for a whole patch is
        forgotten too; the default waits for a box (a whole patch's list is
        never given a default: the user picks a run or Click markers)."""
        self._add_touched.discard("add_marker_source_combo")
        self._add_source_box_key = None
        if self._add_sources_offered_whole:
            return
        was = getattr(self, "_add_applying_preset", False)
        self._add_applying_preset = True        # a default is not a pick
        try:
            self._select_default_add_source()
        finally:
            self._add_applying_preset = was

    def get_add_preset(self) -> str:
        from packages.epicycle_annotator.add_preview import ADD_PRESETS

        idx = int(self.add_preset_combo.currentIndex())
        return ADD_PRESETS[idx] if 0 <= idx < len(ADD_PRESETS) else "nuclei"

    def set_add_preset(self, preset: str, *, silent: bool = False) -> None:
        """Select ``preset`` (by key) and apply its untouched values."""
        from packages.epicycle_annotator.add_preview import ADD_PRESETS

        if preset not in ADD_PRESETS:
            return
        idx = ADD_PRESETS.index(preset)
        if silent:
            self.add_preset_combo.blockSignals(True)
        try:
            self.add_preset_combo.setCurrentIndex(idx)
        finally:
            if silent:
                self.add_preset_combo.blockSignals(False)
        self.apply_add_preset_values(preset)

    def apply_add_preset_values(self, preset: str) -> None:
        """Set every key the user has NOT touched this session to the
        preset's value; never resizes the box; unchecks Place
        markers."""
        values = _ADD_PRESET_VALUES.get(preset, {})
        self._add_applying_preset = True
        try:
            for attr, val in values.items():
                if attr in self._add_touched:
                    continue
                w = getattr(self, attr, None)
                if w is None:
                    continue
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QComboBox):
                    w.setCurrentIndex(int(val))
                else:
                    w.setValue(val)
            if preset == "nuclei":
                self._apply_add_look_for_defaults()
            if (preset == "cells"
                    and "add_marker_source_combo" not in self._add_touched
                    and not self._add_sources_offered_whole):
                # the source the user picked is kept across a switch; a
                # whole patch's list is never given a default (it holds
                # the runs and Click markers)
                self._select_default_add_source()
                if not (self.add_markers_btn.isChecked()
                        or self.add_walls_btn.isChecked()):
                    self.add_markers_btn.setChecked(True)
            if (preset == "cells" and self.add_markers_btn.isChecked()
                    and self.add_walls_btn.isChecked()):
                # Markers was opened on Nuclei while Cells' Walls card was
                # open (hidden there, so left alone): Cells comes back with
                # its Walls card open, one card at a time
                self.add_markers_btn.setChecked(False)
            if preset != "cells" and self.add_look_markers():
                self._add_open_markers_card()
            self.add_place_markers_btn.setChecked(False)
        finally:
            self._add_applying_preset = False
        self._apply_add_visibility()
        self._refresh_add_summaries()

    def _on_add_preset_index_changed(self, _idx: int) -> None:
        preset = self.get_add_preset()
        self.apply_add_preset_values(preset)
        cb = getattr(self, "_on_add_preset_cb", None)
        if cb is not None:
            try:
                cb(preset)
            except Exception as exc:
                print("[ADD] preset callback raised: {!r}".format(exc))

    def set_add_marker_sources(self, entries) -> None:
        """``entries`` = ``[(text, enabled, key)]`` with ``key`` =
        ``(kind, run_id)``; the index -> key map lives beside the combo.
        The full list is kept: on a Cells whole patch the combo offers only
        its "Objects from ..." items and Click markers
        (``_offer_add_sources``)."""
        self._add_source_entries_all = [
            (str(text), bool(enabled), tuple(key))
            for text, enabled, key in (entries or ())]
        combo = self.add_marker_source_combo
        prev = self.get_add_marker_source() if self._add_source_keys else None
        self._fill_add_source_combo(prev)
        if combo.currentIndex() < 0 and not self._add_sources_offered_whole:
            # a default is not the user's choice; a whole patch's list
            # (the runs and Click markers) is never given one
            was = getattr(self, "_add_applying_preset", False)
            self._add_applying_preset = True
            try:
                self._select_default_add_source()
            finally:
                self._add_applying_preset = was
        self._apply_add_visibility()

    def _add_offered_source_entries(self) -> list:
        """The From items the combo offers now: all of them, on a box and
        on a whole patch alike."""
        return list(getattr(self, "_add_source_entries_all", []))

    def _fill_add_source_combo(self, want) -> None:
        """Rebuild the From combo from the offered items with ``want`` (a
        key) current when it is offered and enabled, else blank.  Every
        index change here is silent: a rebuild is never the user's pick."""
        combo = self.add_marker_source_combo
        idx = int(combo.currentIndex())
        old_keys = list(self._add_source_keys)
        before = tuple(old_keys[idx]) if 0 <= idx < len(old_keys) else None
        combo.blockSignals(True)
        try:
            combo.clear()
            self._add_source_keys = []
            for text, enabled, key in self._add_offered_source_entries():
                combo.addItem(text)
                i = combo.count() - 1
                self._add_source_keys.append(tuple(key))
                combo.setItemData(i, text, Qt.ToolTipRole)
                if not enabled:
                    item = combo.model().item(i)
                    if item is not None:
                        item.setEnabled(False)
            target = -1
            if want is not None and tuple(want) in self._add_source_keys:
                j = self._add_source_keys.index(tuple(want))
                if self._add_source_enabled(j):
                    target = j
            combo.setCurrentIndex(target)
        finally:
            combo.blockSignals(False)
        now = self.get_add_marker_source()
        if now != before and now[0] == "labels":
            # another run (or none) is current: the old run's count goes
            self._add_run_markers = None

    def _offer_add_sources(self) -> bool:
        """A Cells whole patch grows its cells from any source a box takes
        (another run's objects, nuclei masks, Click markers).  Rebuilt once
        per change of the whole-patch flag, silently (no source callback,
        nothing marked touched).  Going whole, the pick on the box stays
        picked; a blank combo stays blank (a run is never pre-selected).
        Back on a box a pick made on the whole patch stays; a blank one
        takes the box's default.  True when the list was rebuilt."""
        ui = getattr(self, "_add_ui", {})
        want = bool(ui.get("whole_patch")) and bool(ui.get("box"))
        if want == bool(self._add_sources_offered_whole):
            return False
        idx = int(self.add_marker_source_combo.currentIndex())
        keys = self._add_source_keys
        cur = tuple(keys[idx]) if 0 <= idx < len(keys) else None
        # set before the rebuild: the rebuild reads it
        self._add_sources_offered_whole = want
        if want:
            keep = (cur if cur is not None
                    and (cur[0] != "labels" or cur[1] is not None) else None)
            self._add_source_box_key = None if keep is not None else cur
            self._fill_add_source_combo(keep)
        else:
            back = self._add_source_box_key
            self._add_source_box_key = None
            self._fill_add_source_combo(back if back is not None else cur)
            combo = self.add_marker_source_combo
            if combo.currentIndex() < 0:
                # the pick is gone (dropped or grey): a box never shows a
                # blank From (it would read as clicks) -- the default, and
                # it is not the user's pick
                combo.blockSignals(True)
                try:
                    combo.setCurrentIndex(self._default_add_source_index())
                finally:
                    combo.blockSignals(False)
                if self.get_add_marker_source()[0] != "labels":
                    self._add_run_markers = None
        return True

    def _add_source_enabled(self, idx: int) -> bool:
        try:
            item = self.add_marker_source_combo.model().item(int(idx))
            return item is not None and item.isEnabled()
        except Exception:
            return False

    def _default_add_source_index(self) -> int:
        """Nuclei masks on >= 2 channels, clicks on 1; a run is listed first
        but never pre-selected.  -1 when neither is offered."""
        keys = list(self._add_source_keys)
        for want in ("nuclei_masks", "paint"):
            for i, key in enumerate(keys):
                if key[0] == want and self._add_source_enabled(i):
                    return i
        return -1

    def _select_default_add_source(self) -> None:
        """Nuclei masks on >= 2 channels, clicks on 1; a run is listed first
        but never pre-selected."""
        pick = self._default_add_source_index()
        if pick >= 0:
            self.add_marker_source_combo.setCurrentIndex(pick)

    def _add_auto_sources_grey(self) -> bool:
        """True when every run item and the nuclei item are disabled --
        the membrane-only case the amber line is for."""
        keys = list(getattr(self, "_add_source_keys", []))
        autos = [i for i, k in enumerate(keys)
                 if k[0] in ("labels", "nuclei_masks")]
        return bool(autos) and not any(self._add_source_enabled(i)
                                       for i in autos)

    def get_add_marker_source(self):
        """``(kind, run_id)``.  A blank combo on a Cells whole patch is "a
        run, none picked": ``("labels", None)``."""
        idx = int(self.add_marker_source_combo.currentIndex())
        keys = getattr(self, "_add_source_keys", [])
        if 0 <= idx < len(keys):
            return tuple(keys[idx])
        if getattr(self, "_add_sources_offered_whole", False):
            return ("labels", None)
        return ("paint", None)

    def _on_add_source_index_changed(self, _idx: int) -> None:
        self._add_source_note = ""            # a refusal was about the old one
        # the user's pick: a pick set aside for a whole patch is replaced
        self._add_source_box_key = None
        # the user's pick: kept across preset switches and box ends
        self._mark_add_touched("add_marker_source_combo")
        self.add_place_markers_btn.setChecked(False)
        self._apply_add_visibility()
        self._refresh_add_summaries()
        cb = getattr(self, "_on_add_source_cb", None)
        if cb is not None:
            try:
                cb(self.get_add_marker_source())
            except Exception as exc:
                print("[ADD] source callback raised: {!r}".format(exc))

    # --- Look for nuclei (the Nuclei preset) ------------------------------
    def set_add_look_for_items(self, entries) -> None:
        """``entries`` = ``[(text, enabled, key)]`` with ``key`` =
        ``("inside", run_id)`` (``run_id`` None for the disabled no-run
        item).  Index 0, Anywhere, is always first and is not passed; "At
        the markers I place" (key ``("markers", None)``, always enabled) is
        always last and is not passed either.  The current key is kept when
        it is still offered and enabled; otherwise Anywhere -- a run is
        never pre-selected."""
        from packages.epicycle_annotator import add_preview as AP

        combo = self.add_look_for_combo
        prev = self.get_add_look_for()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(AP.ADD_LOOK_ANYWHERE)
            combo.setItemData(0, AP.ADD_LOOK_ANYWHERE, Qt.ToolTipRole)
            self._add_look_keys = [(AP.LOOK_ANYWHERE, None)]
            self._add_look_nuclei = {}
            for entry in entries or ():
                text, enabled, key = tuple(entry)[:3]
                combo.addItem(str(text))
                i = combo.count() - 1
                self._add_look_keys.append(tuple(key))
                # a 4th entry: that run's target says nuclei
                self._add_look_nuclei[tuple(key)] = bool(
                    tuple(entry)[3]) if len(tuple(entry)) > 3 else False
                # the grey no-run item carries how to get a run (the page
                # does not: the help would sit there on every Nuclei page
                # of a single-run workspace)
                no_run_item = tuple(key)[1] is None
                combo.setItemData(
                    i, AP.ADD_HELP_INSIDE_NO_RUN if no_run_item
                    else str(text), Qt.ToolTipRole)
                if not enabled:
                    item = combo.model().item(i)
                    if item is not None:
                        item.setEnabled(False)
            # the last item: one click per nucleus (a box or a whole
            # patch)
            combo.addItem(AP.ADD_LOOK_MARKERS)
            combo.setItemData(combo.count() - 1, AP.ADD_LOOK_MARKERS,
                              Qt.ToolTipRole)
            self._add_look_keys.append((AP.LOOK_MARKERS, None))
            target = 0
            if prev in self._add_look_keys[1:]:
                j = self._add_look_keys.index(prev)
                if self._add_look_enabled(j):
                    target = j
            combo.setCurrentIndex(target)
        finally:
            combo.blockSignals(False)
        self._sync_add_look_tooltip()
        if self.get_add_look_for() != prev:
            # the run it read is gone: back to Anywhere, said like a change
            self._on_add_look_for_index_changed(combo.currentIndex())
        else:
            self._apply_add_visibility()

    def _sync_add_look_tooltip(self) -> None:
        """The closed combo cuts a run item's text (its name, its target,
        "(reconstruction, not curated)"), so the combo's tooltip is the
        current item's whole text; on Anywhere (nothing cut) it is the
        control's help line."""
        combo = self.add_look_for_combo
        idx = int(combo.currentIndex())
        if idx <= 0:
            combo.setToolTip(ADD_HELP_LINES["add_look_for_combo"])
        else:
            combo.setToolTip(combo.itemText(idx))

    def _add_look_enabled(self, idx: int) -> bool:
        try:
            item = self.add_look_for_combo.model().item(int(idx))
            return item is not None and item.isEnabled()
        except Exception:
            return False

    def get_add_look_for(self):
        """``("anywhere", None)``, ``("inside", run_id)`` or ``("markers",
        None)`` -- by index."""
        from packages.epicycle_annotator.add_preview import LOOK_ANYWHERE

        idx = int(self.add_look_for_combo.currentIndex())
        keys = getattr(self, "_add_look_keys", [])
        if 0 <= idx < len(keys):
            return tuple(keys[idx])
        return (LOOK_ANYWHERE, None)

    def add_look_inside(self) -> bool:
        from packages.epicycle_annotator.add_preview import LOOK_INSIDE

        return (self.get_add_preset() != "cells"
                and self.get_add_look_for()[0] == LOOK_INSIDE)

    def add_look_markers(self) -> bool:
        """The Nuclei preset with Look for nuclei ▸ At the markers I
        place."""
        from packages.epicycle_annotator.add_preview import LOOK_MARKERS

        return (self.get_add_preset() != "cells"
                and self.get_add_look_for()[0] == LOOK_MARKERS)

    def _add_chevron_shown(self, attr: str) -> bool:
        """Is this card's chevron on the page for the preset and Look for
        nuclei now?  Markers: on Cells, and on Nuclei with the markers;
        Walls: on Cells only; Tissue outline (Nuclei's): on Anywhere and the
        markers, on a box or a whole patch, before a result."""
        if self.get_add_preset() == "cells":
            return attr in ("add_markers_btn", "add_walls_btn")
        if attr == "add_nuc_outline_btn":
            return self._add_nuc_outline_offered()
        return attr == "add_markers_btn" and self.add_look_markers()

    def _add_nuc_outline_offered(self) -> bool:
        """Nuclei ▸ Anywhere or the markers, on a box or a whole patch,
        with no candidates up and not in the review: the Tissue outline card
        is on the page (never on Inside each cell: the cells confine the
        nuclei)."""
        ui = getattr(self, "_add_ui", {})
        box = bool(ui.get("box"))
        return (self.get_add_preset() != "cells"
                and not self.add_look_inside()
                and box
                and not bool(ui.get("candidates"))
                and not bool(ui.get("review")))

    def _add_open_markers_card(self) -> None:
        """Open the Markers card (a Cells preset switch does the same when no
        card is open); on Nuclei ▸ the markers it opens whatever else was
        open, and the other cards on the page close."""
        if not self._add_chevron_shown("add_markers_btn"):
            return
        if self.get_add_preset() != "cells":
            if not self.add_markers_btn.isChecked():
                self.add_markers_btn.setChecked(True)
            else:
                self._on_add_subsection_toggled("add_markers_btn", True)
            return
        shown = [a for a in ("add_markers_btn", "add_walls_btn")
                 if self._add_chevron_shown(a)]
        if shown and not any(getattr(self, a).isChecked() for a in shown):
            self.add_markers_btn.setChecked(True)

    def _add_has_other_run(self) -> bool:
        return any(k[0] == "inside" and k[1] is not None
                   for k in getattr(self, "_add_look_keys", []))

    def _apply_add_look_for_defaults(self) -> None:
        """Inside each cell starts with "Smooth the added objects" OFF (its
        recipe already tidies each nucleus); Anywhere has it on -- unless
        the user set it this session."""
        if "add_auto_smooth_cb" in self._add_touched:
            return
        inside = self.add_look_inside()
        was = getattr(self, "_add_applying_preset", False)
        self._add_applying_preset = True
        try:
            self.add_auto_smooth_cb.setChecked(not inside)
        finally:
            self._add_applying_preset = was

    def _on_add_look_for_index_changed(self, _idx: int) -> None:
        self._add_source_note = ""            # a refusal was about the old run
        self._sync_add_look_tooltip()
        self._apply_add_look_for_defaults()
        # a click on Anywhere or Inside must never place a hidden marker:
        # Place markers goes off at every Look-for change (as at a preset
        # or source change)
        self.add_place_markers_btn.setChecked(False)
        if self.add_look_markers():
            # the markers' card opens (a Cells preset switch opens it too)
            self._add_open_markers_card()
        self._apply_add_visibility()
        self._refresh_add_summaries()
        cb = getattr(self, "_on_add_look_for_cb", None)
        if cb is not None:
            try:
                cb(self.get_add_look_for())
            except Exception as exc:
                print("[ADD] look-for callback raised: {!r}".format(exc))

    def _fire_add_whole(self, *_a) -> None:
        cb = getattr(self, "_on_add_whole_cb", None)
        if cb is not None:
            try:
                cb()
            except Exception as exc:
                print("[ADD] whole-patch callback raised: {!r}".format(exc))

    def get_add_wall_params(self) -> dict:
        ticks = {k: getattr(self, "add_prep_{}_cb".format(k)).isChecked()
                 for k in ("zcorrect", "bgsub", "lift", "denoise")}
        idx = int(self.add_wall_kind_combo.currentIndex())
        kind = ADD_WALL_KINDS[idx] if 0 <= idx < len(ADD_WALL_KINDS) else "membrane"
        return {"channel": self.add_wall_channel_name(),
                "chain": chain_from_ticks(ticks, drop_clahe=True),
                "wall_kind": kind}

    def add_nuclei_channel_name(self) -> str:
        c = self.add_nuclei_channel_combo
        return c.currentText() if c.currentIndex() >= 0 else ""

    def add_wall_channel_name(self) -> str:
        c = self.add_wall_channel_combo
        return c.currentText() if c.currentIndex() >= 0 else ""

    def _on_add_threshold_typed(self) -> None:
        """A TYPED threshold (editingFinished) -- the only way the box
        writes back."""
        sb = self.add_threshold_sb
        if (getattr(self, "_add_threshold_is_pending", False)
                and sb.value() == sb.minimum()):
            # focus left a PENDING box without a number typed: nothing set
            return
        v = float(sb.value())
        self._set_add_threshold(v)
        cb = getattr(self, "_on_add_threshold_cb", None)
        if cb is not None:
            try:
                cb(v)
            except Exception:
                pass

    def _on_add_nuclei_threshold_typed(self) -> None:
        cb = getattr(self, "_on_add_nuclei_threshold_cb", None)
        if cb is not None:
            try:
                cb(float(self.add_nuclei_threshold_sb.value()))
            except Exception:
                pass

    def _set_all_add_checks(self, on: bool) -> None:
        lst = self.add_candidate_list
        lst.blockSignals(True)
        try:
            for i in range(lst.count()):
                lst.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        finally:
            lst.blockSignals(False)
        self._on_add_candidate_check_changed()

    # --- Cells: Grow by, the tissue outline, Bridge gaps ---------------------
    @staticmethod
    def _set_add_item_enabled(combo, idx, on) -> None:
        try:
            item = combo.model().item(int(idx))
            if item is not None:
                item.setEnabled(bool(on))
        except Exception:
            pass

    def get_add_grow(self) -> str:
        """``"flood"`` (Fill to the walls) or ``"compete"`` (Spread from
        each marker) -- by index."""
        from packages.epicycle_annotator.add_preview import GROW_KEYS

        idx = int(self.add_grow_combo.currentIndex())
        return GROW_KEYS[idx] if 0 <= idx < len(GROW_KEYS) else GROW_KEYS[0]

    def get_add_outline(self) -> str:
        """``"cleared"`` / ``"shrink"`` / ``"none"`` -- by index."""
        from packages.epicycle_annotator.add_preview import (
            OUTLINE_KEYS,
            OUTLINE_NONE,
        )

        idx = int(self.add_outline_combo.currentIndex())
        return OUTLINE_KEYS[idx] if 0 <= idx < len(OUTLINE_KEYS) else (
            OUTLINE_NONE)

    def set_add_outline(self, key) -> None:
        """Pick ``key`` QUIETLY (the orchestrator's fall-back when the Clears
        behind the cleared-objects outline were undone); not a user choice,
        so the touched set is not changed."""
        from packages.epicycle_annotator.add_preview import OUTLINE_KEYS

        if key not in OUTLINE_KEYS:
            return
        combo = self.add_outline_combo
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(OUTLINE_KEYS.index(key))
        finally:
            combo.blockSignals(False)
        self._apply_add_visibility()
        self._refresh_add_summaries()

    def _on_add_outline_index_changed(self, _idx: int) -> None:
        self._apply_add_visibility()
        self._refresh_add_summaries()
        cb = getattr(self, "_on_add_outline_cb", None)
        if cb is not None:
            try:
                cb(self.get_add_outline())
            except Exception as exc:
                print("[ADD] outline callback raised: {!r}".format(exc))

    # --- Nuclei: the Tissue outline card (an option, off by default) ------
    def get_add_nuc_outline(self) -> str:
        """Nuclei's own Tissue outline: ``"cleared"`` / ``"shrink"`` /
        ``"none"`` -- by index."""
        from packages.epicycle_annotator.add_preview import (
            OUTLINE_KEYS,
            OUTLINE_NONE,
        )

        combo = getattr(self, "add_nuc_outline_combo", None)
        if combo is None:
            return OUTLINE_NONE
        idx = int(combo.currentIndex())
        return OUTLINE_KEYS[idx] if 0 <= idx < len(OUTLINE_KEYS) else (
            OUTLINE_NONE)

    def set_add_nuc_outline(self, key) -> None:
        """Pick ``key`` QUIETLY (the orchestrator's fall-back when the Clears
        behind the cleared-objects outline were undone); not a user choice,
        so the touched set is not changed."""
        from packages.epicycle_annotator.add_preview import OUTLINE_KEYS

        if key not in OUTLINE_KEYS:
            return
        combo = self.add_nuc_outline_combo
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(OUTLINE_KEYS.index(key))
        finally:
            combo.blockSignals(False)
        self._apply_add_visibility()
        self._refresh_add_summaries()

    def _on_add_nuc_outline_index_changed(self, _idx: int) -> None:
        self._apply_add_visibility()
        self._refresh_add_summaries()
        cb = getattr(self, "_on_add_outline_cb", None)
        if cb is not None:
            try:
                cb(self.get_add_nuc_outline())
            except Exception as exc:
                print("[ADD] outline callback raised: {!r}".format(exc))

    def _add_nuc_outline_on(self) -> str:
        """The Nuclei outline asked for now: its pick on Anywhere and the
        markers, on a box or a whole patch; ``"none"`` on Cells and on
        Inside each cell (the orchestrator's rule)."""
        from packages.epicycle_annotator.add_preview import OUTLINE_NONE

        if self.get_add_preset() == "cells" or self.add_look_inside():
            return OUTLINE_NONE
        return self.get_add_nuc_outline()

    def _apply_add_bridge_default(self) -> None:
        """Bridge gaps starts at 10 um, or 16 um on a patch with one channel
        (a count rule: it never guesses which stain that channel is) --
        unless the user set it this session."""
        from packages.epicycle_annotator import add_preview as AP

        if "add_bridge_sb" in self._add_touched:
            return
        single = bool(self._add_ui.get("single_channel"))
        want = AP.ADD_BRIDGE_UM_ONE_CHANNEL if single else AP.ADD_BRIDGE_UM
        sb = self.add_bridge_sb
        if abs(float(sb.value()) - float(want)) > 1e-9:
            sb.blockSignals(True)
            try:
                sb.setValue(float(want))
            finally:
                sb.blockSignals(False)

    def _apply_add_cells_rules(self) -> None:
        """The Cells choices that follow the box and the patch, applied
        QUIETLY (never a user choice):

        * Grow by shows on a box and a whole patch alike, both items
          pickable (on a whole patch Fill is the fill with a background
          marker, or the box rule as you click; Spread is Spread);
        * an install that cannot spread greys Spread from each marker, and
          a pick of it falls back to the fill -- on a whole patch too;
        * "Outline of the objects you cleared" is grey, its reason in its
          words, until a Clear is in force; an untouched Tissue outline
          is the cleared outline when there is one, else the shrink-wrap,
          on a box and a whole patch alike."""
        from packages.epicycle_annotator import add_preview as AP

        ui = self._add_ui
        numba_ok = ui.get("numba_ok") is not False
        in_force = bool(ui.get("cleared_in_force"))
        g = self.add_grow_combo
        g.blockSignals(True)
        try:
            g.setItemText(0, AP.ADD_GROW_FLOOD)
            self._set_add_item_enabled(g, 0, True)
            g.setItemText(1, AP.ADD_GROW_SPREAD if numba_ok
                          else AP.ADD_GROW_SPREAD_GREY)
            self._set_add_item_enabled(g, 1, numba_ok)
            for i in range(g.count()):
                g.setItemData(i, g.itemText(i), Qt.ToolTipRole)
            spread = AP.GROW_KEYS.index(AP.GROW_SPREAD)
            if not numba_ok and g.currentIndex() == spread:
                g.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_FLOOD))
        finally:
            g.blockSignals(False)
        o = self.add_outline_combo
        ci = AP.OUTLINE_KEYS.index(AP.OUTLINE_CLEARED)
        o.blockSignals(True)
        try:
            o.setItemText(ci, AP.ADD_OUTLINE_CLEARED if in_force
                          else AP.ADD_OUTLINE_CLEARED_GREY)
            self._set_add_item_enabled(o, ci, in_force)
            for i in range(o.count()):
                o.setItemData(i, o.itemText(i), Qt.ToolTipRole)
            if "add_outline_combo" not in self._add_touched:
                # one default for every region: the outline of the objects
                # cleared while a Clear is in force, else the shrink-wrap
                want = (AP.OUTLINE_CLEARED if in_force
                        else AP.OUTLINE_SHRINK)
                if want != self.get_add_outline():
                    o.setCurrentIndex(AP.OUTLINE_KEYS.index(want))
        finally:
            o.blockSignals(False)
        # Nuclei's own Tissue outline: the same grey cleared item; its
        # untouched default is always No outline (never picked for the user)
        n = getattr(self, "add_nuc_outline_combo", None)
        if n is None:
            return
        n.blockSignals(True)
        try:
            n.setItemText(ci, AP.ADD_OUTLINE_CLEARED if in_force
                          else AP.ADD_OUTLINE_CLEARED_GREY)
            self._set_add_item_enabled(n, ci, in_force)
            # each item's tooltip says what that choice does (the help line
            # is off the page: the take-off under the 3.1-screen ceiling)
            for i, key in enumerate(AP.OUTLINE_KEYS[:n.count()]):
                n.setItemData(i, AP.nuclei_outline_help(key),
                              Qt.ToolTipRole)
            if ("add_nuc_outline_combo" not in self._add_touched
                    and self.get_add_nuc_outline() != AP.OUTLINE_NONE):
                n.setCurrentIndex(AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        finally:
            n.blockSignals(False)

    def _apply_add_visibility(self) -> None:
        """Every hidden-widget switch of the page, from one place."""
        ui = getattr(self, "_add_ui", {})
        preset = self.get_add_preset()
        cells = preset == "cells"
        box = bool(ui.get("box"))
        review = bool(ui.get("review"))
        cands = bool(ui.get("candidates"))
        kind = self.get_add_marker_source()[0]
        whole = bool(ui.get("whole_patch")) and box
        inside = self.add_look_inside()
        # Nuclei ▸ At the markers I place: the Markers card with the clicks
        # (a box or the whole patch); the histogram folds while the markers
        # are the method
        markers = self.add_look_markers()
        if hasattr(self, "add_grow_combo"):
            self._apply_add_cells_rules()
        numba_ok = ui.get("numba_ok") is not False
        self._add_tune_container.setVisible(not review)
        self._add_review_container.setVisible(review)
        # no box: [Place box] [Whole patch]; a box: S M L Whole Numbers... ⌖.
        # The Whole buttons are never grey (a choice that cannot fill a
        # whole patch greys Preview, and S1 says why)
        self._add_idle_row.setVisible(not box)
        for _b in (self.add_start_btn, self.add_whole_btn):
            _b.setVisible(not box)
        self._add_box_container.setVisible(box)
        from packages.epicycle_annotator import add_preview as AP

        if whole:
            box_help = AP.ADD_HELP_WHOLE
        else:
            box_help = AP.ADD_HELP_DRAG if box else AP.ADD_HELP_BOX_EMPTY
        self._add_box_help.setText(box_help)
        # the whole patch does not move: Recenter is grey
        self.add_recenter_btn.setEnabled(not whole)
        busy = bool(getattr(self, "_add_busy_state", False))
        # hidden while a result is on screen AND while busy (U-LOW)
        self._add_box_help.setVisible(not cands and not (box and busy))
        self._add_brightness_container.setVisible(not cells)
        # Inside each cell reads no threshold: the Brightness card keeps
        # Channel and Look for nuclei, and says what Inside does instead.
        # The histogram shows on Anywhere only: on the markers the clicks
        # pick the nuclei and the threshold only sets their edge; on
        # Anywhere it folds while a Nuclei card (Tissue outline) is on the
        # page AND open
        nuc_offered = (hasattr(self, "add_nuc_outline_btn")
                       and self._add_nuc_outline_offered())
        nuc_card_open = nuc_offered and self.add_nuc_outline_btn.isChecked()
        plot_shown = not inside and not markers and not nuc_card_open
        for w in (self.add_clahe_cb, self._add_threshold_row):
            w.setVisible(not inside)
        for w in (getattr(self, "_add_plot_placeholder", None),
                  getattr(self, "_add_plot_widget", None)):
            if w is not None:
                w.setVisible(plot_shown)
        # a whole patch the page cannot fill measures nothing (the same
        # rule the box-info gather reads): no Auto line says "Measuring..."
        if inside:
            look = AP.LOOK_INSIDE
        elif markers:
            look = AP.LOOK_MARKERS
        else:
            look = AP.LOOK_ANYWHERE
        nothing_measured = whole and AP.whole_measures_nothing(
            preset, kind, look)
        # the Brightness Auto line: measured for a box, or a whole patch
        # with the markers
        if not box:
            threshold_help = AP.ADD_AUTO_NEEDS_BOX
        elif self.add_channel_combo.currentIndex() < 0:
            # no Brightness channel: nothing is measured, so never
            # "Measuring..."
            threshold_help = AP.ADD_HELP_THRESHOLD
        elif self._add_brightness_flat:
            threshold_help = AP.ADD_AUTO_NOTHING_TO_SPLIT
        elif self._add_brightness_pending:
            threshold_help = AP.ADD_AUTO_PENDING
        else:
            threshold_help = AP.ADD_HELP_THRESHOLD
        self._add_threshold_help.setText(threshold_help)
        # the threshold help is hidden while candidates are up (the result
        # card is the news then), on Inside, and on a whole patch that
        # measures nothing (S1 says why Preview is grey); with the
        # histogram folded its own help ("drag the red line") goes too, an
        # Auto status line stays
        folded_help = (not plot_shown
                       and threshold_help == AP.ADD_HELP_THRESHOLD)
        self._add_threshold_help.setVisible(
            not inside and not cands and not nothing_measured
            and not folded_help)
        pre_result = not cands and not review
        note = (getattr(self, "_add_source_note", "")
                or str(ui.get("levels_note") or ""))
        if inside:
            look_key = self.get_add_look_for()
            picked_nuclei = (getattr(self, "_add_look_nuclei", None)
                             or {}).get(look_key, False)
            if look_key[1] is None:
                # Inside with no run (the no-run item is disabled, so only
                # an item that vanished could leave it current): how to
                # get one.  Never shown under Anywhere -- there the words
                # live in the grey item's tooltip
                generic = AP.ADD_HELP_INSIDE_NO_RUN
            elif picked_nuclei:
                generic = AP.ADD_HELP_INSIDE_PICKED_NUCLEI
            else:
                generic = AP.ADD_HELP_INSIDE
            # a refused run's reason first, then a refused channel's
            look_help = note or generic
        else:
            look_help = ""
        self._add_inside_help.setText(look_help)
        self._add_inside_help.setVisible(
            not cells and pre_result and bool(look_help))
        self._add_inside_count.setVisible(
            not cells and inside and pre_result and box)
        derive_note = str(ui.get("derive_note") or "")
        self._add_derive_note.setText(derive_note)
        derive_shown = bool(derive_note) and pre_result
        self._add_derive_note.setVisible(derive_shown)
        if hasattr(self, "add_clear_box_action"):
            # Clear's "mostly inside this box" means nothing on the whole
            # patch: only "Every object in the patch…" is offered
            self.add_clear_box_action.setVisible(not whole)
        # idle Cells shows only the preset and its help; the Markers and
        # Walls chevrons appear once a box exists (mockup 1; U-M5e).  On
        # Nuclei the container holds the Markers card while the markers are
        # the method, on a box or the whole patch
        self._add_cells_container.setVisible(
            (cells and box) or (markers and box) or nuc_offered)
        self._apply_add_card_bodies()
        self._add_preset_help.setText(AP.ADD_HELP_CELLS if cells
                                      else AP.ADD_HELP_NUCLEI)
        # Nuclei ▸ the markers with a marker placed: the preset's help line
        # (what Nuclei finds) gives its room to the clicks' hint line, which
        # stays at every marker count -- the take-off that keeps the
        # markers' states under the 3.1-screen ceiling
        n_markers = max(self.add_click_list.count(),
                        int(getattr(self, "_add_click_n", 0) or 0))
        self._add_preset_help.setVisible(not (markers and n_markers > 0))
        # the From row and what each source shows are Cells' (on Nuclei the
        # markers are the clicks)
        self._add_source_row.setVisible(cells)
        self._add_nuclei_rows.setVisible(cells and kind == "nuclei_masks")
        # the run count only for a picked run (a whole patch's blank combo
        # counts nothing)
        self.add_run_count_label.setVisible(
            cells and kind == "labels"
            and self.get_add_marker_source()[1] is not None)
        clicks = (cells and kind == "paint") or markers
        self._add_click_rows.setVisible(clicks)
        # "Update as I click" under Advanced: wherever the markers are
        # clicks -- Nuclei ▸ At the markers I place and Cells ▸ Click
        # markers (a Cells box over the live limit, and a whole patch, wait
        # for Preview: S1 says so)
        self.add_live_cb.setVisible(clicks)
        # ... and beside it "Always update as I click": Cells ▸ Click
        # markers with the tick on only (hidden, never grey, where it does
        # nothing)
        self.add_live_always_cb.setVisible(
            cells and kind == "paint" and self.add_live_cb.isChecked())
        # ... and between them the reason a click is not run as you click
        # here (Cells ▸ Click markers, the tick on, "Always update as I
        # click" off, a reason pushed)
        live_note = str(ui.get("live_tick_note") or "")
        self._add_live_note.setText(live_note)
        self._add_live_note.setVisible(
            cells and kind == "paint" and self.add_live_cb.isChecked()
            and not self.add_live_always_cb.isChecked() and bool(live_note))
        # the marker list before a result only -- over
        # candidates / REVIEW S5's list is the list; kept, not cleared
        self.add_click_list.setVisible(
            clicks and self.add_click_list.count() > 0
            and not cands and not review)
        if markers:
            # Nuclei: the clicks' hint line, at every marker count (it names
            # Shift+click and Ctrl+Z, which matter once a marker is placed)
            help_text = AP.ADD_HELP_CLICK_KEYS
        elif kind == "nuclei_masks":
            if not self.add_nuclei_channel_name():
                # nothing to measure: "pending" would never end (live-shot
                # fix F9)
                help_text = AP.ADD_AUTO_NEEDS_CHANNEL
            elif not box or nothing_measured:
                # measured for a box only, never for a whole patch
                help_text = AP.ADD_HELP_NUCLEI_MASKS
            elif self._add_nuclei_flat:
                help_text = AP.ADD_AUTO_NOTHING_TO_SPLIT
            elif self._add_nuclei_pending:
                help_text = AP.ADD_AUTO_PENDING
            else:
                help_text = AP.ADD_HELP_NUCLEI_MASKS
        elif kind == "labels":
            help_text = (getattr(self, "_add_source_note", "")
                         or AP.help_run(self.add_core_erode_sb.value()))
        else:
            # Cells ▸ Click markers: the same hint line (the ball's size is
            # beside Place markers)
            help_text = AP.ADD_HELP_CLICK_KEYS
        self._add_source_help.setText(help_text)
        # Cells keeps its help line for every source, as always; on the
        # Nuclei markers the hint line stays whatever the marker count,
        # before and over a result (live, a result is on screen from the
        # first click on, and extending happens over it)
        self._add_source_help.setVisible(cells or markers)
        # only when BOTH automatic sources are grey (U-LOW), and
        # only BEFORE a result: a pre-result instruction, off over
        # candidates and REVIEW -- the take-off that replaced 4.11 step (4)
        # when A3's outline row put 7c 0.5 px over.  Off on a whole patch:
        # Click markers are picked already and the patch's own lines say
        # how they fill; with the marker list and the longest amber the
        # page was 790 px, over its 3.1-screen ceiling
        self._add_membrane_only_label.setVisible(
            cells and kind == "paint" and self._add_auto_sources_grey()
            and not whole and not cands and not review)
        self._add_result_container.setVisible(cands)
        # (Cells' Advanced: below, with Nuclei's shrink-wrap rows)
        self._add_nuclei_adv.setVisible(not cells)
        # Separate touching objects shows on Anywhere and the markers (one
        # tick for both: the markers' on / off); the markers show their own
        # Spacing in place of Anywhere's, and only while the tick is on (it
        # does nothing unticked: hidden, never grey)
        self.add_separate_cb.setVisible(True)
        self._add_spacing_row.setVisible(not markers)
        self._add_marker_spacing_row.setVisible(
            markers and self.add_separate_cb.isChecked())
        # Cells: Grow by on a box and a whole patch, Wall cost and Growth
        # limit with Spread, Include membrane with Spread or on a whole
        # patch filled with a background marker (its line at the top says
        # what fills it under each pick), Bridge gaps with the shrink-wrap;
        # the Walls card's Tissue outline, its help (before a result only)
        # and Show (with an outline chosen)
        outline = self.get_add_outline() if cells else "none"
        by_outline = cells and outline != "none"
        # Nuclei's own Tissue outline (Anywhere and the markers; "none" on
        # Cells and Inside each cell)
        nuc_outline = (self._add_nuc_outline_on()
                       if hasattr(self, "add_nuc_outline_combo") else "none")
        nuc_by_outline = not cells and nuc_outline != "none"
        nuc_shrink = not cells and nuc_outline == "shrink"
        whole_cells = cells and whole
        spread = cells and self.get_add_grow() == AP.GROW_SPREAD
        # the box rule is the fill to the walls only (Spread is never filled
        # as you click)
        live_rule = (whole_cells and bool(ui.get("live_whole_rule"))
                     and not spread)
        # the same slot, Spread's words, the live rule's (as you click, or
        # on Preview where the click fill stepped aside -- S1 says the
        # same) or the background fill's
        self._add_whole_fill_label.setText(AP.whole_fill_help(
            live_rule, ui.get("live_whole_ok"), spread=spread,
            always=(self.add_live_cb.isChecked()
                    and self.add_live_always_cb.isChecked())))
        self._add_whole_fill_label.setVisible(whole_cells)
        self._add_grow_row.setVisible(cells)
        self._add_spread_adv.setVisible(spread)
        self._add_wall_cost_row.setVisible(spread)
        self._add_growth_limit_row.setVisible(spread)
        self.add_include_membrane_cb.setVisible(
            spread or (whole_cells and not live_rule))
        self._add_bridge_row.setVisible((cells and outline == "shrink")
                                        or nuc_shrink)
        self._add_outline_help.setText(AP.outline_help(outline))
        self._add_outline_help.setVisible(cells and not cands and not review)
        self.add_show_outline_cb.setVisible(by_outline)
        # the Show legend: one computation, for the tick on the page
        show_cb = (self.add_show_outline_cb if cells
                   else getattr(self, "add_nuc_show_outline_cb", None))
        shown_by = by_outline if cells else nuc_by_outline
        legend = ""
        if shown_by and show_cb is not None and show_cb.isChecked():
            if ui.get("outline_refused"):
                legend = AP.ADD_SHOW_OUTLINE_REFUSED
            elif ui.get("outline_drawing"):
                legend = AP.ADD_OUTLINE_DRAWING
            elif ui.get("outline_pending") or not ui.get("outline_drawn",
                                                         True):
                legend = AP.ADD_OUTLINE_PENDING
            elif not ui.get("outline_in_side_views", True):
                legend = AP.ADD_SHOW_OUTLINE_MAIN_ONLY
            else:
                legend = AP.ADD_HELP_SHOW_OUTLINE
        cells_legend = legend if cells else ""
        self._add_show_outline_legend.setText(cells_legend)
        self._add_show_outline_legend.setVisible(bool(cells_legend))
        if hasattr(self, "add_nuc_outline_combo"):
            # the Nuclei card's help (its words kept current; the line is
            # off the page, in the items' tooltips), Show with an outline
            # chosen, the legend while Show is ticked
            nuc_legend = "" if cells else legend
            self._add_nuc_outline_help.setText(
                AP.nuclei_outline_help(self.get_add_nuc_outline()))
            self.add_nuc_show_outline_cb.setVisible(nuc_by_outline)
            self._add_nuc_show_outline_legend.setText(nuc_legend)
            self._add_nuc_show_outline_legend.setVisible(bool(nuc_legend))
        # Anywhere's Separate / Spacing / Local contrast strength, or
        # Inside's Nucleus radius / Nucleus edge / Gap that splits two nuclei
        self._add_anywhere_adv.setVisible(not inside)
        self._add_inside_adv.setVisible(inside)
        # the spacing never greys a micron row; at
        # the default (1, 1, 1) the amber note under S1 says so instead --
        # wherever a knob is in microns, and not over candidates or REVIEW
        # (the ceiling states)
        dark_on = self.add_dark_outside_cb.isChecked()
        # D-1: the dark-off leak note, the spacing note's rules (Cells, no
        # candidates, not REVIEW) -- not under a tissue outline, which
        # decides what is outside whatever the tick says
        dark_note = (cells and not dark_on and not by_outline and not review
                     and not cands)
        # one amber slot on the page: a derive note (Cells) takes it, and
        # the dark-off note gives way (its reason, ADD_DARK_OFF, is in the
        # Advanced help slot anyway); on a whole patch no face is a box face
        dark_amber = dark_note and not derive_shown
        # on a whole patch under Spread the growth limit, not "a few µm",
        # is what stops a cell in open background: the Spread note takes
        # the slot (and is not said again in the Advanced help slot)
        spread_amber = dark_amber and whole and spread
        self.add_dark_off_note.setText(
            AP.ADD_NOTE_SPREAD_OPEN if spread_amber
            else AP.ADD_NOTE_DARK_OFF_LEAK_WHOLE if whole
            else AP.ADD_NOTE_DARK_OFF_LEAK)
        self.add_dark_off_note.setVisible(dark_amber)
        # A-M2 (the A1-A4 audit): QC Features -> Settings... is GREY while a
        # box holds the Step 3 lock, so the boxed note says Cancel first
        spacing_text = (AP.ADD_NOTE_DEFAULT_SPACING_BOXED if box
                        else AP.ADD_NOTE_DEFAULT_SPACING)
        self.add_spacing_note.setText(spacing_text)
        # Nothing refuses at the default spacing: Cells (a box or the
        # whole patch), the Nuclei markers (their Ball), Inside each cell
        # (every knob in microns) and a Nuclei Tissue outline (measured in
        # microns) run on it, and this note says so -- on a box or the
        # whole patch alike (Anywhere runs on a whole patch too)
        spacing_note = ((cells or markers or inside or nuc_by_outline)
                        and bool(ui.get("default_spacing"))
                        and not review and not cands)
        # the two amber notes STACKED put 7f at 812 px against 759.5; the
        # dark-off note wins S1's slot and the spacing note moves to the
        # Advanced help slot (751 px).  A derive note (Cells) wins over both:
        # the spacing note moves the same way
        spacing_moved = spacing_note and (dark_amber or derive_shown)
        self.add_spacing_note.setVisible(spacing_note and not spacing_moved)
        # the run source's own Advanced row, only with its source (the
        # ball shows with the click rows)
        self._add_core_erode_row.setVisible(cells and kind == "labels")
        # under a tissue outline the tick and Dark space reach are not used
        # (grey, the reason in the Advanced help slot); Dark below still
        # sets the shrink-wrap
        self.add_dark_outside_cb.setEnabled(not by_outline)
        self.add_dark_clearance_sb.setEnabled(dark_on and not by_outline)
        # Nuclei's shrink-wrap: Dark below is live whatever the (hidden,
        # remembered) dark tick says -- the shrink-wrap is all it sets there
        self.add_dark_factor_sb.setEnabled(dark_on or by_outline
                                           or nuc_shrink)
        self.add_dark_factor_sb.setToolTip(
            AP.ADD_HELP_DARK_BELOW_NUCLEI if nuc_shrink
            else ADD_HELP_LINES.get("add_dark_factor_sb", ""))
        # Cells' Advanced holds Nuclei's two shrink-wrap rows: on Nuclei only
        # Dark below and Bridge gaps show (the tick, Dark space reach and
        # its help are Cells')
        self._add_cells_adv.setVisible(cells or nuc_shrink)
        self.add_dark_outside_cb.setVisible(cells)
        self._add_dark_reach_row.setVisible(cells)
        self._add_dark_below_row.setVisible(cells or nuc_shrink)
        # the grey tick and reach carry the reason in their own tooltip too
        # (the help slot sits at the bottom of Advanced)
        for _name in ("add_dark_outside_cb", "add_dark_clearance_sb"):
            getattr(self, _name).setToolTip(
                AP.ADD_DARK_BY_OUTLINE if by_outline
                else ADD_HELP_LINES.get(_name, ""))
        # the Edge tick and Edge zone mean nothing on a whole patch (no face
        # is a box edge): hidden, not grey (a grey control's reason would
        # sit in a tooltip a disabled widget never shows)
        self.add_clear_border_cb.setVisible(not whole)
        self._add_edge_zone_row.setVisible(not whole)
        # ADD_HELP_DARK describes the two Dark rows; with the tick off they
        # are grey and the help slot carries the reason instead --
        # "ADD_HELP_DARK is not shown" (live-shot fix F3)
        self._add_dark_help.setVisible(cells and dark_on and not by_outline)
        adv_msg = ""
        if by_outline:
            adv_msg = AP.ADD_DARK_BY_OUTLINE
        elif cells and not dark_on and not whole:
            # (a whole patch has no box edges: the amber under S1 says what
            # a cell does in open background there)
            adv_msg = AP.ADD_DARK_OFF
        if spacing_moved:
            adv_msg = "\n\n".join([m for m in (adv_msg, spacing_text) if m])
        extra = []
        if spread and not by_outline and not dark_on and not spread_amber:
            # nothing but the growth limit stops a cell in open background
            # (Spread, on a box or a whole patch -- unless the amber slot
            # says it already)
            extra.append(AP.ADD_NOTE_SPREAD_OPEN)
        if cells and not numba_ok and not whole:
            # on a whole patch S1 says it, and Preview is grey
            extra.append(AP.no_numba_sentence(ui.get("numba_why")))
        if extra:
            adv_msg = "\n\n".join([m for m in [adv_msg] + extra if m])
        self._add_advanced_help.setText(adv_msg)
        self._add_advanced_help.setVisible(bool(adv_msg))
        # Advanced waits for a box (both presets), and is not in REVIEW
        adv_shown = box and not review
        self._add_advanced_toggle.setVisible(adv_shown)
        self._add_advanced_container.setVisible(
            adv_shown and self._add_advanced_toggle.isChecked())
        self.add_preview_btn.setText("Preview again" if cands else "Preview")
        self.add_clear_btn.setVisible(box and not cands and not review)
        self._apply_add_idle_hint()
        # the footer line is always on screen: it says what the two
        # buttons do in every state (over a result too)
        self._add_footer_summary.setVisible(True)
        self.add_accept_btn.setToolTip(AP.ADD_DONE_TIP if review else "")
        if not cells:
            self.add_show_walls_cb.setEnabled(False)

    def _refresh_add_summaries(self) -> None:
        """The chevron summaries, one elided line each."""
        from packages.epicycle_annotator import add_preview as AP

        cells = self.get_add_preset() == "cells"
        markers = self.add_look_markers()
        kind = self.get_add_marker_source()[0]
        if markers:
            # the Nuclei markers are clicks: "Markers: clicks · 3 placed"
            kind = AP.SOURCE_CLICKS
        ticks = [PREP_TICK_LABELS[k].lower() for k in
                 ("zcorrect", "bgsub", "lift", "denoise")
                 if getattr(self, "add_prep_{}_cb".format(k)).isChecked()]
        # the mockup's summaries, as named constants (U-M5c)
        self._set_chevron_title(self.add_markers_btn, AP.markers_summary(
            kind, channel=self.add_nuclei_channel_name(),
            n_clicks=getattr(self, "_add_click_n", 0),
            n_ticked=(getattr(self, "_add_click_ticked", None)
                      if len(getattr(self, "_add_click_rows_view", []))
                      == getattr(self, "_add_click_n", 0) else None),
            n_run_markers=getattr(self, "_add_run_markers", None)))
        # the ROLE word, not the layer name (mockup; B3.4)
        self._set_chevron_title(self.add_walls_btn, AP.walls_summary(
            channel=(AP.ADD_SUM_WALLS_ROLE if self.add_wall_channel_name()
                     else ""),
            wall_kind=self.add_wall_kind_combo.currentText(),
            prep=ticks,
            outline=(self.get_add_outline() if cells
                     and hasattr(self, "add_outline_combo") else "")))
        self._set_chevron_title(
            self.add_wall_prep_btn,
            "Prepare the channel first: {}".format(
                ", ".join(ticks) if ticks else "none"))
        if hasattr(self, "add_nuc_outline_btn"):
            self._set_chevron_title(
                self.add_nuc_outline_btn,
                AP.nuclei_outline_summary(self.get_add_nuc_outline()))
        if cells:
            parts = []
            ui = getattr(self, "_add_ui", {})
            if (ui.get("whole_patch") and ui.get("box")
                    and self.get_add_grow() != AP.GROW_SPREAD):
                # a whole patch on Fill is filled to the walls (with a
                # background marker); filled as you click, the membrane
                # tick does not apply.  Spread: its own summary, below
                parts.append(AP.advanced_fill_summary(
                    self.add_include_membrane_cb.isChecked(),
                    live=bool(ui.get("live_whole_rule"))))
            elif self.get_add_grow() == AP.GROW_SPREAD:
                parts.append(AP.advanced_grow_summary(
                    self.add_wall_cost_sb.value(),
                    self.add_growth_limit_sb.value(),
                    self.add_include_membrane_cb.isChecked()))
            # only the rows on screen: the shrink with another run's
            # objects (the ball is on the Markers card, not in Advanced)
            if kind == "labels":
                parts.append("shrink {:.1f} µm".format(
                    self.add_core_erode_sb.value()))
            outline = self.get_add_outline()
            if outline == AP.OUTLINE_SHRINK:
                # the tissue outline decides what is outside
                parts.append("bridge {:g} µm".format(
                    self.add_bridge_sb.value()))
            elif outline == AP.OUTLINE_CLEARED:
                pass
            elif self.add_dark_outside_cb.isChecked():
                parts.append("dark {:.1f} µm".format(
                    self.add_dark_clearance_sb.value()))
            else:
                parts.append("dark off")
        elif self.add_look_inside():
            # the two shares with two decimals, as their boxes show them
            # ("gap 0.70", not "gap 0.7")
            parts = [AP.ADD_SUM_ADVANCED_INSIDE.format(
                r="{:g}".format(self.add_nuc_radius_sb.value()),
                e="{:.2f}".format(self.add_nuc_edge_sb.value()),
                g="{:.2f}".format(self.add_two_gap_sb.value()))]
        elif markers and not self.add_separate_cb.isChecked():
            # the markers unticked: their Spacing is hidden and unused
            parts = ["one per blob"]
        else:
            # Anywhere and the markers: the shared tick, and the Spacing on
            # screen (the markers' own box on the markers)
            spacing_sb = (self.add_marker_spacing_sb if markers
                          else self.add_min_distance_sb)
            parts = ["separate touching" if self.add_separate_cb.isChecked()
                     else "one per blob",
                     "spacing {}".format(spacing_sb.value())]
        if (not cells and hasattr(self, "add_nuc_outline_combo")
                and self._add_nuc_outline_on() == AP.OUTLINE_SHRINK):
            # Nuclei's shrink-wrap: Bridge gaps is on screen under Advanced
            parts.append("bridge {:g} µm".format(self.add_bridge_sb.value()))
        if self.add_auto_smooth_cb.isChecked():
            parts.append("smooth {:.1f}".format(
                self.add_smooth_sigma_sb.value()))
        self._set_chevron_title(self._add_advanced_toggle,
                                "Advanced: " + " · ".join(parts))

    # --- Add callback / state API ---------------------------------------------

    def set_add_callbacks(
        self,
        *,
        on_start=None,
        on_box_changed=None,
        on_recenter=None,
        on_size_preset=None,
        on_clahe_toggled=None,
        on_threshold_changed=None,
        on_otsu_reset=None,
        on_preview=None,
        on_accept=None,
        on_reject=None,
        on_advanced_changed=None,
        on_select_piece=None,
        on_refine_piece=None,
        on_merge_pieces=None,
        on_batch_refine=None,
        on_smooth_pieces=None,
        on_candidate_changed=None,
        on_add_another=None,
        on_undo_all=None,
        on_preset_changed=None,
        on_marker_source_changed=None,
        on_place_markers_toggled=None,
        on_click_undo=None,
        on_click_clear=None,
        on_stop=None,
        on_show_walls=None,
        on_all=None,
        on_none=None,
        on_numbers=None,
        on_nuclei_auto=None,
        on_nuclei_threshold=None,
        on_channels_changed=None,
        on_dark_outside_toggled=None,
        on_contour_toggled=None,
        on_click_tick=None,
        on_clear=None,
        on_look_for_changed=None,
        on_whole_patch=None,
        on_outline_changed=None,
        on_show_outline=None,
        on_live_toggled=None,
        on_click_pick=None,
        on_live_always_toggled=None,
    ):
        """Wire the Add page's controls to external orchestration callbacks.

        Mirrors ``set_split_callbacks`` (disconnect-before-connect idiom).
        """
        def _wire(btn_attr, cb, signal="clicked"):
            # Disconnect-before-connect, of the slot THIS method wired last
            # time only: a bare disconnect() would also cut the page's own
            # slots on the same signal (the touched set, the summaries).
            if cb and hasattr(self, btn_attr):
                sig = getattr(getattr(self, btn_attr), signal)
                reg = getattr(self, "_add_wired", None)
                if reg is None:
                    reg = {}
                    self._add_wired = reg
                old = reg.get((btn_attr, signal))
                if old is not None:
                    try:
                        sig.disconnect(old)
                    except (TypeError, RuntimeError):
                        pass
                sig.connect(cb)
                reg[(btn_attr, signal)] = cb

        _wire("add_start_btn", on_start)
        _wire("add_recenter_btn", on_recenter)
        _wire("add_preview_btn", on_preview)
        _wire("add_accept_btn", on_accept)
        _wire("add_reject_btn", on_reject)
        _wire("add_otsu_btn", on_otsu_reset)
        _wire("add_refine_piece_btn", on_refine_piece)
        _wire("add_batch_refine_btn", on_batch_refine)
        _wire("add_merge_pieces_btn", on_merge_pieces)
        _wire("add_again_btn", on_add_another)
        # on_undo_all is kept (the setters are additive only) and wires
        # nothing: REVIEW's Undo all is the footer's secondary now, which
        # on_reject routes by state
        _wire("add_clahe_cb", on_clahe_toggled, signal="toggled")
        _wire("add_stop_btn", on_stop)
        _wire("add_click_undo_btn", on_click_undo)
        _wire("add_click_clear_btn", on_click_clear)
        _wire("add_nuclei_auto_btn", on_nuclei_auto)
        _wire("add_place_markers_btn", on_place_markers_toggled,
              signal="toggled")
        _wire("add_show_walls_cb", on_show_walls, signal="toggled")

        if on_box_changed is not None:
            self._on_add_box_changed_cb = on_box_changed
        if on_click_tick is not None:
            # cb(marker_id, ticked), the id from the row's UserRole
            self._on_add_click_tick_cb = on_click_tick
        if on_click_pick is not None:
            # cb(object_id): a click on a marker or result row (not on its
            # tick) makes that object current
            self._on_add_click_pick_cb = on_click_pick
        if on_clear is not None:
            # cb("box" | "patch"); the two menu actions are connected once,
            # at build, to bound methods that read this attribute
            self._on_add_clear_cb = on_clear
        if on_look_for_changed is not None:
            # cb(("anywhere", None) | ("inside", run_id)); the combo is
            # connected once, at build
            self._on_add_look_for_cb = on_look_for_changed
        if on_whole_patch is not None:
            # cb(); both Whole patch buttons are connected once, at build
            self._on_add_whole_cb = on_whole_patch
        if on_outline_changed is not None:
            # cb("cleared" | "shrink" | "none"); both presets' combos are
            # connected once, at build, and call it with their own pick
            self._on_add_outline_cb = on_outline_changed
        _wire("add_show_outline_cb", on_show_outline, signal="toggled")
        _wire("add_nuc_show_outline_cb", on_show_outline, signal="toggled")
        if on_size_preset is not None:
            self._on_add_size_preset_cb = on_size_preset
        if on_candidate_changed is not None:
            self._on_add_candidate_changed_cb = on_candidate_changed
        if on_threshold_changed is not None:
            self._on_add_threshold_cb = on_threshold_changed
        if on_preset_changed is not None:
            self._on_add_preset_cb = on_preset_changed
        if on_marker_source_changed is not None:
            self._on_add_source_cb = on_marker_source_changed
        if on_numbers is not None:
            self._on_add_numbers_cb = on_numbers
        if on_nuclei_threshold is not None:
            self._on_add_nuclei_threshold_cb = on_nuclei_threshold
        if on_all is not None:
            _wire("add_all_btn", on_all)
        if on_none is not None:
            _wire("add_none_btn", on_none)
        if on_channels_changed is not None:
            for attr in ("add_channel_combo", "add_nuclei_channel_combo",
                         "add_wall_channel_combo"):
                _wire(attr, lambda *_a: on_channels_changed(),
                      signal="currentIndexChanged")
        _wire("add_dark_outside_cb", on_dark_outside_toggled,
              signal="toggled")
        _wire("add_contour_cb", on_contour_toggled, signal="toggled")
        _wire("add_live_cb", on_live_toggled, signal="toggled")
        _wire("add_live_always_cb", on_live_always_toggled, signal="toggled")
        if on_advanced_changed is not None:
            self._on_add_advanced_cb = on_advanced_changed
            # connected ONCE: the slot reads the callback attribute, so a
            # second wiring only replaces the attribute
            if not getattr(self, "_add_adv_signals_wired", False):
                self._add_adv_signals_wired = True
                for attr, sig in _ADD_ADVANCED_SIGNALS:
                    w = getattr(self, attr, None)
                    if w is not None:
                        getattr(w, sig).connect(
                            lambda *_: self._on_add_advanced_changed()
                        )
        if on_select_piece is not None and hasattr(self, "add_pieces_list"):
            self._on_add_select_piece_cb = on_select_piece
            try:
                self.add_pieces_list.itemSelectionChanged.disconnect()
            except TypeError:
                pass
            self.add_pieces_list.itemSelectionChanged.connect(
                self._on_add_piece_selection_changed
            )

    def get_add_params(self) -> dict:
        """Snapshot the current Add controls for the orchestrator.  Inside
        each cell adds its own keys (``look_for``, ``nuc_radius_um``,
        ``nuc_edge``, ``two_gap``); At the markers I place adds ``look_for``
        and the markers' own Spacing (``marker_spacing``, in voxels of the
        full-size image); Anywhere's snapshot is exactly what it always
        was."""
        params = {
            "channel": self.add_channel_combo.currentText()
            if hasattr(self, "add_channel_combo") else "",
            "clahe": bool(self.add_clahe_cb.isChecked())
            if hasattr(self, "add_clahe_cb") else False,
            "clahe_clip": float(self.add_clahe_clip_sb.value())
            if hasattr(self, "add_clahe_clip_sb") else 0.01,
            "min_size": int(self.add_min_size_sb.value())
            if hasattr(self, "add_min_size_sb") else 0,
            "separate": bool(self.add_separate_cb.isChecked())
            if hasattr(self, "add_separate_cb") else True,
            "min_distance": int(self.add_min_distance_sb.value())
            if hasattr(self, "add_min_distance_sb") else 3,
            "clear_border": bool(self.add_clear_border_cb.isChecked())
            if hasattr(self, "add_clear_border_cb") else False,
            "edge_margin": int(self.add_edge_margin_sb.value())
            if hasattr(self, "add_edge_margin_sb") else 3,
            "auto_smooth": bool(self.add_auto_smooth_cb.isChecked())
            if hasattr(self, "add_auto_smooth_cb") else False,
            "smooth_sigma": float(self.add_smooth_sigma_sb.value())
            if hasattr(self, "add_smooth_sigma_sb") else 1.5,
            "preset": self.get_add_preset()
            if hasattr(self, "add_preset_combo") else "nuclei",
            "click_radius_um": float(self.add_click_radius_sb.value())
            if hasattr(self, "add_click_radius_sb") else 1.0,
            "core_erode_um": float(self.add_core_erode_sb.value())
            if hasattr(self, "add_core_erode_sb") else 1.0,
            "dark_clearance_um": float(self.add_dark_clearance_sb.value())
            if hasattr(self, "add_dark_clearance_sb") else 2.0,
            "dark_otsu_factor": float(self.add_dark_factor_sb.value())
            if hasattr(self, "add_dark_factor_sb") else 0.5,
            "nuclei_threshold": float(self.add_nuclei_threshold_sb.value())
            if hasattr(self, "add_nuclei_threshold_sb") else 0.0,
            "dark_outside": bool(self.add_dark_outside_cb.isChecked())
            if hasattr(self, "add_dark_outside_cb") else True,
        }
        if (getattr(self, "add_grow_combo", None) is not None
                and self.get_add_preset() == "cells"):
            # how cells grow and the tissue outline (Cells only: the Nuclei
            # snapshot is exactly what it always was)
            params.update(
                growth=self.get_add_grow(),
                wall_cost=float(self.add_wall_cost_sb.value()),
                growth_limit=float(self.add_growth_limit_sb.value()),
                include_membrane=bool(self.add_include_membrane_cb.isChecked()),
                outline=self.get_add_outline(),
                bridge_um=float(self.add_bridge_sb.value()))
        if getattr(self, "add_look_for_combo", None) is not None \
                and self.add_look_inside():
            from packages.epicycle_annotator.add_preview import LOOK_INSIDE

            params.update(
                look_for=LOOK_INSIDE,
                nuc_radius_um=float(self.add_nuc_radius_sb.value()),
                nuc_edge=float(self.add_nuc_edge_sb.value()),
                two_gap=float(self.add_two_gap_sb.value()))
        elif getattr(self, "add_look_for_combo", None) is not None \
                and self.add_look_markers():
            # At the markers I place: two keys more, Look for and the
            # markers' own Spacing (the Ball and the shared tick are
            # already in the snapshot)
            from packages.epicycle_annotator.add_preview import LOOK_MARKERS

            params.update(look_for=LOOK_MARKERS)
            if hasattr(self, "add_marker_spacing_sb"):
                params.update(marker_spacing=int(
                    self.add_marker_spacing_sb.value()))
        if (getattr(self, "add_nuc_outline_combo", None) is not None
                and self.get_add_preset() != "cells"):
            # Nuclei's Tissue outline, only when one is chosen (Anywhere and
            # the markers; never on Inside each cell): exactly two keys more
            nuc_outline = self._add_nuc_outline_on()
            if nuc_outline != "none":
                params.update(outline=nuc_outline,
                              bridge_um=float(self.add_bridge_sb.value()))
        return params

    def _on_add_box_slider_changed(self, *_args) -> None:
        """Fire on_box_changed with the 6 slider values (cz,cy,cx,sz,sy,sx)."""
        if getattr(self, "_add_box_loading", False):
            return
        cb = getattr(self, "_on_add_box_changed_cb", None)
        if cb is None or not getattr(self, "_add_box_sliders", None):
            return
        s = self._add_box_sliders
        try:
            cb(
                int(s["cz"].value()), int(s["cy"].value()), int(s["cx"].value()),
                int(s["sz"].value()), int(s["sy"].value()), int(s["sx"].value()),
            )
        except Exception:
            pass

    def _on_add_advanced_changed(self) -> None:
        cb = getattr(self, "_on_add_advanced_cb", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def _on_add_piece_selection_changed(self) -> None:
        """Fire on_select_piece on EVERY selection change -- with the id
        when exactly one row is selected, else ``None`` -- so S1 follows
        the selection (B2 fix pass U-M3); toggle Refine/Merge."""
        sel = self.add_pieces_list.selectedItems()
        n = len(sel)
        if hasattr(self, "add_refine_piece_btn"):
            self.add_refine_piece_btn.setEnabled(n == 1)
        if hasattr(self, "add_merge_pieces_btn"):
            self.add_merge_pieces_btn.setEnabled(n == 2)
        pid = sel[0].data(Qt.UserRole) if n == 1 else None
        cb = getattr(self, "_on_add_select_piece_cb", None)
        if cb is not None:
            try:
                cb(int(pid) if pid is not None else None)
            except Exception:
                pass

    def get_selected_add_pieces(self) -> list:
        """The piece IDs currently selected in the REVIEW list."""
        out = []
        if hasattr(self, "add_pieces_list"):
            for it in self.add_pieces_list.selectedItems():
                pid = it.data(Qt.UserRole)
                if pid is not None:
                    out.append(int(pid))
        return out

    def refresh_add_image_combo(self, image_names: list[str],
                                colours: dict | None = None) -> None:
        """Fill the three Add channel combos from the viewer's image layers.

        All three remember the name last chosen in ``_epicycle_channel_choice``
        (not ``currentText()``), so a deliberately blank combo stays blank
        across refreshes; the nuclei and wall combos START blank -- no
        silent pick of layer 0 -- and S1 asks for them by name.

        ``colours`` -- ``{name: (r, g, b)}`` in 0..1, each layer's own
        colour in the views: its item carries a small swatch of it.  The
        item's text stays the layer name."""
        from qtpy.QtCore import QSize
        from qtpy.QtGui import QColor

        icons = {}
        for name, rgb in (colours or {}).items():
            try:
                r, g, b = (min(1.0, max(0.0, float(v))) for v in rgb[:3])
                icons[name] = self._colour_swatch_icon(
                    QColor.fromRgbF(r, g, b))
            except Exception:  # noqa: BLE001 - a swatch is optional
                continue
        for attr, first_fill_blank in (("add_channel_combo", False),
                                       ("add_nuclei_channel_combo", True),
                                       ("add_wall_channel_combo", True)):
            combo = getattr(self, attr, None)
            if combo is None:
                continue
            prev = self._remember_channel_choice(combo)
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(list(image_names))
                if icons:
                    combo.setIconSize(QSize(12, 12))
                    for i, name in enumerate(image_names):
                        if name in icons:
                            combo.setItemIcon(i, icons[name])
                self._restore_channel_choice(combo, prev)
                if not prev and first_fill_blank:
                    combo.setCurrentIndex(-1)
            finally:
                combo.blockSignals(False)

    def update_add_ui_state(
        self,
        *,
        box_active: bool | None = None,
        box_shape: tuple | None = None,
        box_center: tuple | None = None,
        box_size: tuple | None = None,
        histogram_data: dict | None = None,
        threshold_value: float | None = None,
        results_text: str | None = None,
        preview_enabled: bool | None = None,
        accept_enabled: bool | None = None,
        accept_count: int | None = None,
        reject_enabled: bool | None = None,
        start_enabled: bool | None = None,
        status_text: str | None = None,
        review_active: bool | None = None,
        piece_ids: list | None = None,
        piece_sizes: dict | None = None,
        candidate_ids: list | None = None,
        candidate_sizes: dict | None = None,
        candidate_edge_ids: set | None = None,
        marker_enabled: bool | None = None,
        add_batch_refine_enabled: bool | None = None,
        # ---  (all keyword-only, all None) ---
        preset: str | None = None,
        marker_sources: list | None = None,
        click_count: int | None = None,
        place_markers_on: bool | None = None,
        run_count: tuple | None = None,
        nuclei_auto: float | None = None,
        busy: bool | None = None,
        stop_enabled: bool | None = None,
        stage_text: str | None = None,
        measuring: str | None = None,
        candidate_small_ids: set | None = None,
        candidate_flags: dict | None = None,
        notes: list | None = None,
        walls_toggle_enabled: bool | None = None,
        walls_toggle_reason: str | None = None,
        walls_kind: str | None = None,
        box_over_limit: bool | None = None,
        voxel_size_known: bool | None = None,
        box_line: str | None = None,
        footer_text: str | None = None,
        review_voxels: int | None = None,
        review_has_selection: bool | None = None,
        single_channel: bool | None = None,
        status_kind: str | None = None,
        brightness_auto: float | None = None,
        threshold_pending: bool | None = None,
        brightness_flat: bool | None = None,
        nuclei_flat: bool | None = None,
        default_spacing: bool | None = None,
        contour_on: bool | None = None,
        click_markers: list | None = None,
        click_absorbed_ids: list | None = None,
        # --- Inside each cell / Whole patch (keyword-only, None) ---
        look_for_items: list | None = None,
        inside_count: object | None = None,
        derive_note: str | None = None,
        whole_patch: bool | None = None,
        candidate_tips: dict | None = None,
        piece_chips: dict | None = None,
        piece_tips: dict | None = None,
        progress: tuple | None = None,
        levels_note: str | None = None,
        # --- Cells: the tissue outline, the growth engine (keyword-only) ---
        cleared_in_force: bool | None = None,
        outline_pending: bool | None = None,
        numba_ok: bool | None = None,
        numba_why: str | None = None,
        live_whole_rule: bool | None = None,
        live_whole_ok: bool | None = None,
        live_tick_note: str | None = None,
        outline_drawn: bool | None = None,
        outline_refused: bool | None = None,
        outline_drawing: bool | None = None,
        outline_in_side_views: bool | None = None,
        # --- Nuclei ▸ At the markers I place (keyword-only, None) ---
        click_absorbed_tips: dict | None = None,
        # --- Update as I click: the Accept ticks kept by id, and Accept
        # waiting for the last click to be drawn ---
        candidate_checked: set | None = None,
        accept_waiting: bool | None = None,
        # --- extending an object: rows per object, the current one ---
        click_status: dict | None = None,
        click_current: int | None = None,
        candidate_seeds: dict | None = None,
        candidate_status: dict | None = None,
    ) -> None:
        """Update the Add page from the orchestrator.

        Every keyword is None-guarded so partial updates are cheap.
        ``results_text`` and ``marker_enabled`` are accepted and ignored
        (the results line merged into S1; the Boundary stub is gone) --
        the setters are frozen as additive-only.  A PENDING run count is
        ``run_count=()`` and a pending Auto ``nuclei_auto=float("nan")``
        (the Brightness card's ``brightness_auto`` alike), because ``None``
        means "not passed".  ``histogram_data`` is either precomputed bins
        from the box-info worker (``{"counts", "edges"}``; empty lists =
        pending, the plot cleared) or raw ``{"values", "n_bins"}``.
        ``brightness_flat`` / ``nuclei_flat`` say the measurement of that
        Auto ENDED: ``True`` -- the crop is flat, nothing to split;
        ``False`` -- no answer came (a failed or cancelled measurement), so
        nothing is pending any more.  ``measuring`` is the strip's line
        while a reader runs and nothing else does (``""`` none): the bar
        moves, the controls stay live and Stop grey.
        """
        ui = self._add_ui
        if click_current is not None:
            # the current object's id (0: none) -- its rows are bold
            self._add_click_current = int(click_current)
        if candidate_seeds is not None:
            # {id: n}: the result rows of objects grown from several markers
            self._add_candidate_seeds = {int(k): int(v) for k, v in
                                         dict(candidate_seeds).items()}
        if candidate_status is not None:
            # {id: [words]}: what those objects' seeds made (chips)
            self._add_candidate_status = {int(k): list(v) for k, v in
                                          dict(candidate_status).items()}
        if click_status is not None:
            # {id: [words]}: the marker rows' status chips
            self._add_click_status = {int(k): list(v) for k, v in
                                      dict(click_status).items()}
        if candidate_small_ids is not None:
            self._add_candidate_small_ids = {int(i) for i in candidate_small_ids}
        if candidate_flags is not None:
            # {id: ("two", "cut")} -- the check-this chips (never unticked)
            self._add_candidate_flags = {int(k): tuple(v) for k, v in
                                         dict(candidate_flags).items()}
        if candidate_tips is not None:
            self._add_candidate_tips = {int(k): str(v) for k, v in
                                        dict(candidate_tips).items()}
        if look_for_items is not None:
            self.set_add_look_for_items(look_for_items)
        if whole_patch is not None:
            # before the count: its words say "patch" on the whole patch
            ui["whole_patch"] = bool(whole_patch)
        if inside_count is not None:
            from packages.epicycle_annotator.add_preview import (
                inside_count_line,
            )

            # ``()`` = pending (None means "not passed")
            self._add_inside_count.setText(inside_count_line(
                None if inside_count == () else int(inside_count),
                whole_patch=bool(ui.get("whole_patch"))))
        if derive_note is not None:
            ui["derive_note"] = str(derive_note)
        if levels_note is not None:
            # why the patch's levels refused the Brightness channel ("" =
            # none): shown under Look for nuclei on Inside, after a refused
            # run's own reason
            ui["levels_note"] = str(levels_note)
        if cleared_in_force is not None:
            # a Clear still in force on this patch: the cleared-objects
            # outline is offered
            ui["cleared_in_force"] = bool(cleared_in_force)
        if outline_pending is not None:
            ui["outline_pending"] = bool(outline_pending)
        if numba_ok is not None:
            # False: Spread from each marker cannot run in this install
            ui["numba_ok"] = bool(numba_ok)
        if numba_why is not None:
            # why not: "missing" (numba is not installed) or "cache" (its
            # cache folder cannot be written); "" = it can run
            ui["numba_why"] = str(numba_why)
        if live_whole_rule is not None:
            # a Cells whole patch filled as you click, the way a box is:
            # the help line in the Advanced slot says so, the membrane tick
            # (which that fill never reads) is hidden, the summary has no
            # membrane part
            ui["live_whole_rule"] = bool(live_whole_rule)
        if live_whole_ok is not None:
            # ... and whether it is run as you click now (False: the click
            # fill stepped aside; the help line then says Preview fills it)
            ui["live_whole_ok"] = bool(live_whole_ok)
        if live_tick_note is not None:
            # the line under "Update as I click": why a Cells click is not
            # run as you click here ("" none)
            ui["live_tick_note"] = str(live_tick_note)
        if outline_drawn is not None:
            # the outline's skin is drawn now (main and side views)
            ui["outline_drawn"] = bool(outline_drawn)
        if outline_refused is not None:
            # the outline could not be measured (S1 says why)
            ui["outline_refused"] = bool(outline_refused)
        if outline_drawing is not None:
            # the outline's layers are being added right now
            ui["outline_drawing"] = bool(outline_drawing)
        if outline_in_side_views is not None:
            # False: the side views could not take it (the legend says so)
            ui["outline_in_side_views"] = bool(outline_in_side_views)
        if progress is not None:
            self._set_add_progress(progress)
        if box_active is not None:
            ui["box"] = bool(box_active)
            self._set_add_s1_kind("ready" if box_active else "idle")
        # ``voxel_size_known`` is accepted and IGNORED (the spacing
        # never greys anything; the setter is additive-only
        # and frozen, like ``results_text``); ``default_spacing`` drives the
        # amber note under S1
        if default_spacing is not None:
            ui["default_spacing"] = bool(default_spacing)
        if contour_on is not None:
            # the orchestrator's reset: silent -- not a user tick
            self.add_contour_cb.blockSignals(True)
            self.add_contour_cb.setChecked(bool(contour_on))
            self.add_contour_cb.blockSignals(False)
        if single_channel is not None:
            ui["single_channel"] = bool(single_channel)
            self._apply_add_bridge_default()
        if preset is not None:
            self.set_add_preset(preset, silent=True)
        if marker_sources is not None:
            self.set_add_marker_sources(marker_sources)
        # Sync slider ranges + values without firing on_box_changed.
        if (box_shape is not None or box_center is not None
                or box_size is not None) and getattr(self, "_add_box_sliders", None):
            self._add_box_loading = True
            try:
                s = self._add_box_sliders
                if box_shape is not None:
                    Z, Y, X = (int(v) for v in box_shape)
                    s["cz"].setRange(0, max(0, Z - 1))
                    s["cy"].setRange(0, max(0, Y - 1))
                    s["cx"].setRange(0, max(0, X - 1))
                    s["sz"].setRange(1, max(1, Z))
                    s["sy"].setRange(1, max(1, Y))
                    s["sx"].setRange(1, max(1, X))
                if box_center is not None:
                    cz, cy, cx = (int(v) for v in box_center)
                    s["cz"].setValue(cz)
                    s["cy"].setValue(cy)
                    s["cx"].setValue(cx)
                if box_size is not None:
                    sz, sy, sx = (int(v) for v in box_size)
                    s["sz"].setValue(sz)
                    s["sy"].setValue(sy)
                    s["sx"].setValue(sx)
            finally:
                self._add_box_loading = False
        if histogram_data is not None:
            self._update_add_histogram(histogram_data)
        if threshold_pending:
            # final fix pass B4: the stored threshold is None (its value
            # space changed) -- the box shows pending, never the old number
            self._show_add_threshold_pending()
        if threshold_value is not None:
            self._set_add_threshold(threshold_value)
        if preview_enabled is not None:
            self.add_preview_btn.setEnabled(bool(preview_enabled))
        if accept_enabled is not None:
            self.add_accept_btn.setEnabled(bool(accept_enabled))
        if accept_count is not None:
            self._add_accept_text(int(accept_count))
        if reject_enabled is not None:
            # wanted, and live unless a preview runs (Stop is the one exit)
            self._add_reject_wanted = bool(reject_enabled)
            self.add_reject_btn.setEnabled(
                bool(reject_enabled)
                and not bool(getattr(self, "_add_busy_state", False)))
        if start_enabled is not None:
            self.add_start_btn.setEnabled(bool(start_enabled))
        if status_text is not None:
            self.add_status_label.setText(str(status_text))
        if click_absorbed_tips is not None:
            # {id: why}: the other markers the last Preview made nothing
            # from (on stain below the threshold, too small) -- their rows
            # dim, the reason in the tooltip
            self._add_click_tips = {int(k): str(v) for k, v in
                                    dict(click_absorbed_tips).items()}
        if click_absorbed_ids is not None or click_absorbed_tips is not None:
            if click_absorbed_ids is not None:
                # the A5 data-flow audit (B-M1): the markers the last
                # Preview found inside kept objects -- their rows dim, with
                # a tooltip
                self._add_click_absorbed = {int(i)
                                            for i in click_absorbed_ids}
            if click_markers is None:
                lst = self.add_click_list
                lst.blockSignals(True)
                try:
                    for i in range(lst.count()):
                        self._style_add_click_row(lst.item(i))
                finally:
                    lst.blockSignals(False)
        if click_markers is not None:
            # the orchestrator's marker records, as a view
            # -- ``[(id, (z, y, x), ticked), ...]``, one row per object
            self._set_add_click_markers(click_markers)
            # ... and the result list's rows take the current object and
            # the statuses of the same push (the marker rows are rebuilt)
            self._restyle_add_object_rows()
        elif (click_status is not None or click_current is not None
              or candidate_status is not None):
            # a status or the current object alone: the rows restyled
            self._restyle_add_object_rows()
        if click_count is not None:
            from packages.epicycle_annotator.add_preview import (
                click_count_text,
            )

            self._add_click_n = int(click_count)
            view = getattr(self, "_add_click_rows_view", [])
            n_ticked = (getattr(self, "_add_click_ticked", None)
                        if len(view) == self._add_click_n else None)
            self.add_click_count_label.setText(
                click_count_text(self._add_click_n, n_ticked=n_ticked))
        if place_markers_on is not None:
            self.add_place_markers_btn.blockSignals(True)
            self.add_place_markers_btn.setChecked(bool(place_markers_on))
            self.add_place_markers_btn.blockSignals(False)
        if run_count is not None:
            from packages.epicycle_annotator.add_preview import (
                run_count_line,
                run_count_pending_line,
            )

            if run_count:
                self.add_run_count_label.setText(run_count_line(
                    *run_count, whole_patch=bool(ui.get("whole_patch"))))
                self._add_run_markers = int(run_count[1])
            else:
                self.add_run_count_label.setText(run_count_pending_line(
                    whole_patch=bool(ui.get("whole_patch"))))
                self._add_run_markers = None
        # the two Auto lines keep flags here; _apply_add_visibility writes
        # their words (a line with nothing being measured never says
        # "Measuring...": no box, a whole patch that measures nothing, a
        # flat crop, a measurement that ended without an answer)
        if nuclei_auto is not None:
            import math

            v = float(nuclei_auto)
            self._add_nuclei_auto_value = None if math.isnan(v) else v
            self.add_nuclei_auto_btn.setEnabled(not math.isnan(v))
            self._add_nuclei_pending = math.isnan(v)
            self._add_nuclei_flat = False
        if nuclei_flat is not None:
            self._add_nuclei_pending = False
            self._add_nuclei_flat = bool(nuclei_flat)
            if nuclei_flat:
                self.add_nuclei_auto_btn.setEnabled(False)
        if brightness_auto is not None:
            # B4.2: the Brightness Auto is measured on the box-info worker;
            # grey, with the pending sentence in the card's help slot, until
            # it lands
            import math

            pending = math.isnan(float(brightness_auto))
            self.add_otsu_btn.setEnabled(not pending)
            self._add_brightness_pending = pending
            self._add_brightness_flat = False
        if brightness_flat is not None:
            # final fix pass B2: bars landed but no Auto -- a flat box has
            # nothing for Otsu to split; say so instead of pending for ever
            self._add_brightness_pending = False
            self._add_brightness_flat = bool(brightness_flat)
            if brightness_flat:
                self.add_otsu_btn.setEnabled(False)
        # TUNE <-> REVIEW swap.
        if review_active is not None:
            ui["review"] = bool(review_active)
            if review_active or candidate_ids is None:
                self._fill_add_candidates([], {}, set())
            # REVIEW: the primary is Done (it closes the session and KEEPS
            # the objects -- the accept colour), the secondary Undo all
            # (destructive -- the reject colour).  Out of REVIEW: Accept
            # and Close box / Discard
            self.add_reject_btn.setText(
                "Undo all" if review_active else "Close box")
            self.add_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)
            if review_active:
                from packages.epicycle_annotator.add_preview import (
                    ADD_DONE_TIP,
                    ADD_UNDO_ALL_TIP,
                )

                self._add_reject_wanted = True
                self.add_reject_btn.setEnabled(True)
                self.add_reject_btn.setToolTip(ADD_UNDO_ALL_TIP)
                self.add_accept_btn.setText("Done")
                self.add_accept_btn.setEnabled(True)
                self.add_accept_btn.setToolTip(ADD_DONE_TIP)
            else:
                # Close box / Discard: the tooltip follows the footer line
                # (_refresh_add_footer_summary)
                if self.add_accept_btn.text() == "Done":
                    self.add_accept_btn.setText("Accept")
                    self.add_accept_btn.setEnabled(False)
                    self.add_accept_btn.setToolTip("")
            self._set_add_s1_kind("review" if review_active else "idle")
        if piece_ids is not None:
            sizes = {int(k): int(v) for k, v in (piece_sizes or {}).items()}
            # the accepted objects keep their check-this chips in REVIEW
            pchips = {int(k): list(v) for k, v in (piece_chips or {}).items()}
            ptips = {int(k): str(v) for k, v in (piece_tips or {}).items()}
            lst = self.add_pieces_list
            lst.blockSignals(True)
            lst.clear()
            for pid in piece_ids:
                pid = int(pid)
                sz = sizes.get(pid)
                text = "Object {}".format(pid)
                if sz is not None:
                    text += "  ·  {:,}".format(sz)
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, pid)
                item.setData(ADD_CHIPS_ROLE, pchips.get(pid, []))
                item.setIcon(self._seed_swatch_icon(pid))
                tip = text
                if ptips.get(pid):
                    tip += "\n" + ptips[pid]
                item.setToolTip(tip)
                lst.addItem(item)
            lst.blockSignals(False)
            self._cap_add_list(lst)
            self.add_refine_piece_btn.setEnabled(False)
            self.add_merge_pieces_btn.setEnabled(False)
            if add_batch_refine_enabled is None:
                add_batch_refine_enabled = bool(piece_ids)
            from packages.epicycle_annotator.add_preview import ADD_HELP_REVIEW

            self._add_review_help.setText(
                ADD_HELP_REVIEW.format(n=len(piece_ids)))
        if add_batch_refine_enabled is not None:
            self.add_batch_refine_btn.setVisible(bool(add_batch_refine_enabled))
            self.add_batch_refine_btn.setEnabled(bool(add_batch_refine_enabled))
        if review_has_selection is not None:
            self.add_refine_piece_btn.setEnabled(bool(review_has_selection))
        if accept_waiting is not None:
            # "Accepting…" on the button while Accept waits; its count back
            # after
            self._add_accept_waiting = bool(accept_waiting)
            k = len(self.get_checked_add_pieces())
            self._add_accept_text(k)
            if not self._add_ui.get("review"):
                # grey while it waits; live again after when rows are ticked
                self.add_accept_btn.setEnabled(
                    not self._add_accept_waiting and k > 0)
        if candidate_ids is not None:
            self._fill_add_candidates(candidate_ids, candidate_sizes or {},
                                      candidate_edge_ids or set(),
                                      checked=candidate_checked)
        if walls_toggle_enabled is not None or walls_toggle_reason is not None:
            enabled = bool(walls_toggle_enabled)
            self.add_show_walls_cb.setEnabled(enabled)
            reason = walls_toggle_reason or ""
            if not enabled:
                self.add_show_walls_cb.blockSignals(True)
                self.add_show_walls_cb.setChecked(False)
                self.add_show_walls_cb.blockSignals(False)
            self._add_walls_reason = reason
            self._refresh_add_walls_legend()
        self._apply_add_rebuild_state(
            busy=busy, stop_enabled=stop_enabled, stage_text=stage_text,
            notes=notes, footer_text=footer_text, measuring=measuring)
        if status_kind is not None:
            # what S1 MEANS (add_preview.s1_kind): a blocking reason is not
            # a green dot (live-shot fix V7); last, so it wins over the
            # dots the box / review / candidate pushes set above
            self._set_add_s1_kind(str(status_kind))
        if whole_patch is not None or box_active is not None:
            # the From list of a Cells whole patch: rebuilt here, once both
            # flags of this push are applied, and silently
            self._offer_add_sources()
        self._apply_add_visibility()
        self._refresh_add_summaries()
        self._refresh_add_footer_summary()

    def _set_add_click_markers(self, markers) -> None:
        """ (A5): the marker records the orchestrator owns --
        ``[(id, (z, y, x), ticked), ...]`` -- drawn as the marker list (a
        VIEW: rebuilt whole, under ``blockSignals``, so a push never
        reports a tick back)."""
        from packages.epicycle_annotator.add_preview import (
            ADD_MARKER_LIST_ROWS,
            object_row_text,
        )

        # one row per OBJECT: the seeds of one object (markers sharing an
        # id) are one row -- its first seed's place, or its seed count --
        # and its tick (every seed of an object carries the same one)
        rows, at = [], {}
        for i, zyx, t in (markers or []):
            i = int(i)
            if i in at:
                k = at[i]
                rows[k] = rows[k][:3] + (rows[k][3] + 1,)
                continue
            at[i] = len(rows)
            rows.append((i, tuple(int(v) for v in zyx), bool(t), 1))
        self._add_click_rows_view = rows
        self._add_click_ticked = sum(1 for r in rows if r[2])
        lst = self.add_click_list
        lst.blockSignals(True)
        try:
            lst.clear()
            for mid, zyx, ticked, n in rows:
                item = QListWidgetItem(object_row_text(mid, zyx, n))
                item.setData(Qt.UserRole, mid)
                item.setIcon(self._seed_swatch_icon(mid))
                item.setToolTip(ADD_HELP_LINES["add_click_list"])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if ticked else Qt.Unchecked)
                self._style_add_click_row(item)
                lst.addItem(item)
        finally:
            lst.blockSignals(False)
        self._cap_add_list(lst, rows=ADD_MARKER_LIST_ROWS)

    def _style_add_object_row(self, item, status) -> None:
        """A row's status chips (what its seeds made at the last result)
        and its weight: the current object's row is bold.  The status
        words come BEFORE the edge / small chips (chips drop right to left,
        so the status draws first); their long form is a line of the row's
        tooltip."""
        from packages.epicycle_annotator.add_preview import (
            ADD_ROW_STATUS_LONG,
            ADD_ROW_STATUS_TIP,
            status_tip,
        )

        mid = item.data(Qt.UserRole)
        words = [] if mid is None else list(status.get(int(mid), []))
        chips = list(item.data(ADD_CHIPS_ROLE) or [])
        own = set(ADD_ROW_STATUS_LONG)
        item.setData(ADD_CHIPS_ROLE,
                     words + [c for c in chips if c not in own])
        head = ADD_ROW_STATUS_TIP.split("{")[0]
        lines = [ln for ln in str(item.toolTip() or "").split("\n")
                 if ln and not ln.startswith(head)]
        extra = status_tip(words)
        if extra:
            lines.append(extra)
        item.setToolTip("\n".join(lines))
        cur = int(getattr(self, "_add_click_current", 0) or 0)
        font = item.font()
        bold = bool(cur) and mid is not None and int(mid) == cur
        if font.bold() != bold:
            font.setBold(bold)
            item.setFont(font)

    def _restyle_add_object_rows(self) -> None:
        """Both lists' rows take the status words and the bold row again
        (quietly: no tick is reported)."""
        for lst, status in (
                (self.add_click_list,
                 getattr(self, "_add_click_status", None) or {}),
                (self.add_candidate_list,
                 getattr(self, "_add_candidate_status", None) or {})):
            lst.blockSignals(True)
            try:
                for i in range(lst.count()):
                    self._style_add_object_row(lst.item(i), status)
            finally:
                lst.blockSignals(False)

    def _on_add_row_pressed(self, item) -> None:
        """A press on a row: its tick as it was, so the release can tell a
        click on the tick box (the tick changed) from a click on the row."""
        self._add_row_press = (item, None if item is None
                               else item.checkState())

    def _on_add_row_clicked(self, item) -> None:
        """A click on a marker or result row makes its object current (the
        orchestrator decides; an id no marker makes is ignored there).  A
        click on the row's tick box ticks, and picks nothing."""
        press = getattr(self, "_add_row_press", None)
        self._add_row_press = None
        if item is None:
            return
        if (press is not None and press[0] is item
                and press[1] != item.checkState()):
            return
        mid = item.data(Qt.UserRole)
        cb = getattr(self, "_on_add_click_pick_cb", None)
        if cb is not None and mid is not None:
            try:
                cb(int(mid))
            except Exception as exc:
                print(f"[ADD] row pick callback raised: {exc!r}")

    def _style_add_click_row(self, item) -> None:
        """An unticked marker row is drawn dim (its ball is off the view);
        so is a marker the last Preview found inside a kept object, whose
        tooltip then says so (B-M1) -- and a marker the last Preview made
        nothing from for another reason, its tooltip the reason."""
        from qtpy.QtGui import QBrush, QColor

        from packages.epicycle_annotator.add_preview import (
            ADD_MARKER_ABSORBED_TIP,
        )

        on = item.checkState() == Qt.Checked
        mid = item.data(Qt.UserRole)
        absorbed = (mid is not None and int(mid) in
                    getattr(self, "_add_click_absorbed", set()))
        why = (None if mid is None else
               (getattr(self, "_add_click_tips", None) or {}).get(int(mid)))
        marked = absorbed or bool(why)
        item.setForeground(QBrush(QColor(
            "#eeeeee" if on and not marked else "#777777")))
        if absorbed:
            tip = ADD_MARKER_ABSORBED_TIP
        elif why:
            tip = why
        else:
            tip = ADD_HELP_LINES["add_click_list"]
        item.setToolTip(tip)
        # the status chips, their tooltip line and the bold row, after the
        # tooltip (it adds its line to it)
        self._style_add_object_row(
            item, getattr(self, "_add_click_status", None) or {})

    def _on_add_click_list_changed(self, item) -> None:
        """A USER tick in the marker list: restyle the row QUIETLY (a
        restyle inside an itemChanged handler re-enters it, so a quiet restyle
        with signals blocked), recount, and report ``(id, ticked)`` to the orchestrator,
        which owns the tick."""
        from packages.epicycle_annotator.add_preview import click_count_text

        if item is None:
            return
        lst = self.add_click_list
        lst.blockSignals(True)
        try:
            self._style_add_click_row(item)
        finally:
            lst.blockSignals(False)
        mid = item.data(Qt.UserRole)
        ticked = item.checkState() == Qt.Checked
        rows = [(i, p, ticked if i == int(mid) else t, n)
                for i, p, t, n in getattr(self, "_add_click_rows_view", [])]
        self._add_click_rows_view = rows
        self._add_click_ticked = sum(1 for r in rows if r[2])
        self.add_click_count_label.setText(click_count_text(
            len(rows), n_ticked=self._add_click_ticked))
        self._refresh_add_summaries()
        cb = getattr(self, "_on_add_click_tick_cb", None)
        if cb is not None and mid is not None:
            try:
                cb(int(mid), bool(ticked))
            except Exception as exc:
                print(f"[ADD] marker tick callback raised: {exc!r}")

    def _fill_add_candidates(self, candidate_ids, candidate_sizes,
                             candidate_edge_ids, checked=None) -> None:
        """The candidate rows: "Object {id}", the chips (two in a cell, cut
        at cell edge, edge, small), the size right-aligned mono; a row with
        an edge or small chip starts unticked -- the two check-this chips
        never untick a row.  ``checked`` (a set of ids, live runs): every
        row takes exactly that tick instead -- the orchestrator keeps the
        user's ticks by id across results."""
        from packages.epicycle_annotator.add_preview import (
            candidate_chips,
            candidate_row_text,
        )

        lst = self.add_candidate_list
        seeds = dict(getattr(self, "_add_candidate_seeds", None) or {})
        status = dict(getattr(self, "_add_candidate_status", None) or {})
        csizes = {int(k): int(v) for k, v in dict(candidate_sizes).items()}
        cedge = {int(i) for i in candidate_edge_ids}
        small = set(getattr(self, "_add_candidate_small_ids", set()))
        flags = dict(getattr(self, "_add_candidate_flags", {}) or {})
        tips = dict(getattr(self, "_add_candidate_tips", {}) or {})
        keep = None if checked is None else {int(i) for i in checked}
        lst.blockSignals(True)
        lst.clear()
        for pid in candidate_ids:
            pid = int(pid)
            sz = csizes.get(pid)
            chips = candidate_chips(pid, edge_ids=cedge, small_ids=small,
                                    flags=flags)
            unticked = [c for c in chips if c in ("edge", "small")]
            text = candidate_row_text(pid, sz, seeds.get(pid, 1))
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, pid)
            item.setData(ADD_CHIPS_ROLE, chips)
            item.setIcon(self._seed_swatch_icon(pid))
            tip = text
            if unticked:
                tip += " (" + ", ".join(unticked) + " -- starts unticked)"
            if tips.get(pid):
                tip += "\n" + tips[pid]
            item.setToolTip(tip)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = (pid in keep) if keep is not None else not unticked
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            # the seeds' status words after the chips; the current object
            # bold
            self._style_add_object_row(item, status)
            lst.addItem(item)
        lst.blockSignals(False)
        self._cap_add_list(lst)
        n = len(candidate_ids)
        self._add_ui["candidates"] = n > 0
        if not self._add_ui.get("review"):
            # REVIEW keeps its Undo all (an empty fill comes with it)
            self.add_reject_btn.setText("Discard" if n > 0 else "Close box")
        self._set_add_s1_kind("review" if n else "ready")
        self._on_add_candidate_check_changed()

    def _refresh_add_walls_legend(self, *_a) -> None:
        """The legend slot: every grey reason ON SCREEN; the legend itself
        while the tick is on (Qt shows no tooltip on a disabled box)."""
        reason = getattr(self, "_add_walls_reason", "")
        if reason:
            self.add_walls_legend.setText(reason)
            self.add_walls_legend.setVisible(True)
        elif self.add_show_walls_cb.isChecked():
            self.add_walls_legend.setText(WALLS_LEGEND_IMAGE)
            self.add_walls_legend.setVisible(True)
        else:
            self.add_walls_legend.setVisible(False)

    def _cap_add_list(self, lst, rows=None) -> None:
        """At most ``ADD_LIST_ROWS`` rows visible (``rows`` for another
        cap -- the marker list's 3); the rest scroll."""
        from packages.epicycle_annotator.add_preview import ADD_LIST_ROWS

        cap = ADD_LIST_ROWS if rows is None else int(rows)
        rows = max(1, min(lst.count(), cap))
        h = lst.sizeHintForRow(0) if lst.count() else 18
        h = max(16, int(h))
        frame = 2 * lst.frameWidth() + 2
        lst.setFixedHeight(rows * h + frame)

    def _apply_add_rebuild_state(self, **kw) -> None:
        """The strip and the notes; None-guarded."""
        busy = kw.get("busy")
        if kw.get("measuring") is not None:
            self._add_measuring_text = str(kw["measuring"])
        if busy is not None or kw.get("measuring") is not None:
            if busy is not None:
                self._add_busy_state = bool(busy)
            b = bool(getattr(self, "_add_busy_state", False))
            # a reader's words while nothing else runs; busy wins
            measuring = ("" if b else
                         str(getattr(self, "_add_measuring_text", "") or ""))
            self._set_add_strip_busy(b, measuring)
            if busy is not None:
                self._set_add_busy_enabled(b)
            if not b:
                # idle: the line is cleared or takes the reader's words (the
                # visibility pass puts the Clear line back)
                self._set_add_stage_line(measuring
                                         or self._add_idle_stage_text())
            elif self._add_stage_full_text == self._add_idle_hint():
                # busy: the idle line never shows lit
                self._set_add_stage_line("")
        if kw.get("stop_enabled") is not None:
            self.add_stop_btn.setEnabled(bool(kw["stop_enabled"]))
        if kw.get("stage_text") is not None:
            self._set_add_stage_line(kw["stage_text"])
        if kw.get("footer_text") is not None:
            self._add_footer_override = kw["footer_text"] or None
        if kw.get("notes") is not None:
            texts = [t for _k, t, _a in kw["notes"] if t]
            full = " ".join(texts)
            self._add_notes_full = full
            self.add_notes_label.setToolTip(full)
            self.add_notes_label.setVisible(bool(texts))
            self._cap_add_notes()

    @staticmethod
    def _add_idle_hint() -> str:
        from packages.epicycle_annotator.add_preview import (
            ADD_STRIP_CLEAR_HINT,
        )

        return ADD_STRIP_CLEAR_HINT

    def _apply_add_idle_hint(self) -> None:
        """The idle strip's stage line says what Clear objects is for while
        the button is on screen and nothing runs (dimmed, as the idle strip
        is).  Other words on the line keep it; the hint comes back when
        they end.  The strip's height is fixed, so this costs no space."""
        lb = getattr(self, "add_stage_label", None)
        btn = getattr(self, "add_clear_btn", None)
        if lb is None or btn is None:
            return
        if getattr(self, "_add_busy_state", False):
            return
        hint = self._add_idle_hint()
        full = str(getattr(self, "_add_stage_full_text", "") or "")
        if full not in ("", hint):
            return                      # other words own the line
        want = "" if btn.isHidden() else hint
        self._set_add_stage_line(want)

    def _add_accept_text(self, k: int) -> None:
        """ "Accept (k)" -- the ticked count -- whenever candidates are on
        screen, "(0)" included; plain "Accept" otherwise (mockup; live-shot
        fix V4)."""
        if getattr(self, "_add_ui", {}).get("review"):
            return                    # REVIEW's primary reads Done
        if getattr(self, "_add_accept_waiting", False):
            from packages.epicycle_annotator.add_preview import (
                ADD_ACCEPT_WAITING_BTN,
            )

            self.add_accept_btn.setText(ADD_ACCEPT_WAITING_BTN)
            return
        cands = bool(getattr(self, "_add_ui", {}).get("candidates"))             and self.add_candidate_list.count() > 0
        self.add_accept_btn.setText(
            "Accept ({})".format(int(k)) if cands else "Accept")

    def _set_add_strip_busy(self, busy: bool, measuring: str = "") -> None:
        """The pinned strip: Stop, the bar and the stage line are ALWAYS
        drawn (mockup); idle they are dimmed and the bar is still, busy the
        bar runs its marquee and Stop is live (live-shot fix V2).
        ``measuring`` (a reader's words, not busy): the bar moves and the
        line is lit, Stop stays grey."""
        self._add_busy_container.setVisible(True)
        self.add_stop_btn.setVisible(True)
        self.add_stage_label.setVisible(True)
        prog = getattr(self, "_add_progress", None)
        moving = bool(busy) or bool(measuring)
        if busy and prog:
            # a whole patch: step i of n (a determinate bar)
            self.add_busy_bar.setRange(0, int(prog[1]))
            self.add_busy_bar.setValue(int(prog[0]))
        elif busy:
            self.add_busy_bar.setRange(0, 0)
        else:
            # not busy: no step kept, Stop grey; still and empty unless a
            # reader runs
            self._add_progress = None
            self.add_stop_btn.setEnabled(False)
            if moving:
                self.add_busy_bar.setRange(0, 0)
            else:
                self.add_busy_bar.setRange(0, 1)
                self.add_busy_bar.setValue(0)
        self.add_busy_bar.setEnabled(moving)
        self.add_stage_label.setEnabled(moving)

    def _set_add_stage_line(self, text) -> None:
        """The strip's one grey line: its words, the full text as its
        tooltip, elided to the width.  Every write goes through here."""
        self._add_stage_full_text = str(text or "")
        self.add_stage_label.setToolTip(self._add_stage_full_text)
        self._apply_add_stage_elision()

    def _add_idle_stage_text(self) -> str:
        """The words of the strip's line while nothing runs or is measured
        (empty: the line is blank).  A reader's words take the line while
        they last, and these come back after."""
        return ""

    def paint_add_wait(self, *, status_text, status_kind="busy",
                       stage_text="", step=None, of=None,
                       stop_enabled=None) -> None:
        """Before a wait on the main thread: S1 and its dot, the strip's
        line and, with ``step`` of ``of``, the bar (step k of n, never full)
        are set directly and painted NOW.  The app is frozen until the wait
        ends, so no queued repaint, layout request or animation would show
        them: S1's row and then the page that holds it are laid out by hand
        (only the page's layout can give S1 another line) and the page
        repainted, then the strip; the event loop is never run here.
        Without a step the bar is left as it is.  ``stop_enabled`` (None:
        left as it is) sets Stop before the paint, so the frozen frame shows
        it as it is (Stop sits in the painted strip).  The next page update
        puts the strip back."""
        self.add_status_label.setText(str(status_text))
        self._set_add_s1_kind(str(status_kind))
        self._set_add_stage_line(stage_text)
        self.add_stage_label.setEnabled(True)
        if stop_enabled is not None:
            self.add_stop_btn.setEnabled(bool(stop_enabled))
        if step is not None and of:
            n = max(1, int(of))
            k = max(0, min(int(step), n - 1))
            self._add_progress = (k, n)
            self.add_busy_bar.setRange(0, n)
            self.add_busy_bar.setValue(k)
            self.add_busy_bar.setEnabled(True)
        try:
            s1 = getattr(self, "_add_s1_container", None)
            if s1 is not None:
                s1.layout().activate()
                page = s1.parentWidget()
                if page is not None and page.layout() is not None:
                    page.layout().activate()
                (page or s1).repaint()
            s4 = getattr(self, "_add_s4_container", None)
            if s4 is not None:
                s4.layout().activate()
                s4.repaint()
        except RuntimeError:              # a torn-down page
            pass

    def _set_add_progress(self, progress) -> None:
        """``(i, n)``: the bar shows step i of n while busy; ``()`` puts it
        back to the moving marquee."""
        prog = tuple(progress or ())
        if len(prog) == 2 and int(prog[1]) > 0:
            self._add_progress = (max(0, min(int(prog[0]), int(prog[1]))),
                                  int(prog[1]))
        else:
            self._add_progress = None
        if getattr(self, "_add_busy_state", False):
            self._set_add_strip_busy(True)

    def _cap_add_notes(self) -> None:
        """The counts note wraps to at most ``ADD_NOTES_LINES`` lines;
        beyond that the last line is elided and the full text rides in the
        tooltip.  The label is exactly that
        many lines tall, top-aligned."""
        from packages.epicycle_annotator.add_preview import ADD_NOTES_LINES

        lb = self.add_notes_label
        fm = lb.fontMetrics()
        m = lb.contentsMargins()
        h = fm.lineSpacing() * int(ADD_NOTES_LINES) + m.top() + m.bottom()
        lb.setFixedHeight(h)
        self._apply_add_notes_elision()

    def _apply_add_notes_elision(self) -> None:
        """Greedy word wrap into ``ADD_NOTES_LINES`` lines at the label's
        own width; the last line elided with "…" when the text runs on."""
        from packages.epicycle_annotator.add_preview import ADD_NOTES_LINES

        lb = getattr(self, "add_notes_label", None)
        if lb is None:
            return
        full = str(getattr(self, "_add_notes_full", "") or "")
        width = lb.contentsRect().width()
        if width <= 0:
            width = 280
        lb.setText(wrap_and_elide(full, lb.fontMetrics(), width,
                                  int(ADD_NOTES_LINES)))

    def _set_add_busy_enabled(self, busy: bool) -> None:
        """``busy`` disables S2, S3, Advanced, Place markers and the
        Numbers... sliders -- disabled, not hidden."""
        for name in ("_add_s2_container", "_add_s3_container",
                     "_add_advanced_container", "_add_advanced_toggle",
                     "_add_result_container"):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(not busy)
        for sl in (getattr(self, "_add_box_sliders", None) or {}).values():
            sl.setEnabled(not busy)
        clear_btn = getattr(self, "add_clear_btn", None)
        if clear_btn is not None:
            clear_btn.setEnabled(not busy)
        reject_btn = getattr(self, "add_reject_btn", None)
        if reject_btn is not None:
            # Close box is grey while a preview runs: Stop is the one exit
            reject_btn.setEnabled(
                not busy and bool(getattr(self, "_add_reject_wanted", False)))

    def _fire_add_clear(self, mode: str) -> None:
        cb = getattr(self, "_on_add_clear_cb", None)
        if cb is not None:
            try:
                cb(mode)
            except Exception as exc:  # noqa: BLE001
                print(f"[ADD-CLEAR] clear callback raised: {exc!r}")

    def _fire_add_clear_box(self, *_a) -> None:
        self._fire_add_clear("box")

    def _fire_add_clear_patch(self, *_a) -> None:
        self._fire_add_clear("patch")

    def _fire_add_size_preset(self, frac: float) -> None:
        """A box-size preset button (S/M/L) → tell the orchestrator to resize."""
        cb = getattr(self, "_on_add_size_preset_cb", None)
        if cb is not None:
            try:
                cb(frac)
            except Exception:
                pass

    def get_checked_add_pieces(self) -> list:
        """Candidate piece IDs currently CHECKED (what Accept will commit)."""
        out = []
        if hasattr(self, "add_candidate_list"):
            for i in range(self.add_candidate_list.count()):
                it = self.add_candidate_list.item(i)
                if it.checkState() == Qt.Checked:
                    pid = it.data(Qt.UserRole)
                    if pid is not None:
                        out.append(int(pid))
        return out

    def _on_add_candidate_check_changed(self, *args) -> None:
        """A candidate checkbox toggled → update the Accept (K) button + notify
        the orchestrator (so it can re-mask the 3D candidate overlay — wired in
        the live-sync pass; a no-op until then)."""
        k = len(self.get_checked_add_pieces())
        if hasattr(self, "add_accept_btn") and not getattr(
                self, "_add_ui", {}).get("review"):
            # (REVIEW's primary is Done, always live)
            self._add_accept_text(k)
            # grey while Accept waits for the last click to be drawn
            self.add_accept_btn.setEnabled(
                k > 0 and not getattr(self, "_add_accept_waiting", False))
        cb = getattr(self, "_on_add_candidate_changed_cb", None)
        if cb is not None:
            try:
                cb()
            except Exception:
                pass

    def release_plots(self) -> None:
        """Let the lazily built pyqtgraph plots (the Add in-box histogram,
        the auto-detect one) go ON THEIR OWN, before this panel is destroyed
        by the annotator's close or relaunch.

        It does NOT cover napari's own close: a deletion posted from inside
        napari's close is never delivered, so a plot still alive then stays
        in the window tree (the session ends release the Add plot while the
        event loop runs, which is what keeps that close clean).  Measured:
        with the Add plot alive, hidden in the floating panel, at napari's
        close, the close aborted ("QObject: shared QObject was deleted
        directly"); other pyqtgraph plots that live until the close do not,
        so the exact double delete is not pinned (a plot inside a FLOATING
        dock that napari re-docks in its close, and pyqtgraph's parentless
        menus, are the leads).  Each plot is unparented (it stays hidden)
        and scheduled for deletion; the panel forgets it, its threshold line
        and its bars, and builds none afterwards (no placeholder is put
        back).  Idempotent; never raises.
        """
        self._release_plot("_add_plot_widget",
                           ("_add_thresh_line", "_add_bar_item"))
        self._release_plot("_detect_plot_widget",
                           ("_detect_thresh_line", "_detect_bar_item"))
        self._detect_last_edt_values = []

    def release_detect_plot(self) -> None:
        """Let the auto-detect histogram go at the end of a Reshape
        session, as ``release_add_plot`` does for Add (measured: a Split
        Detect, Cancel, then napari's close aborted while it was alive).  A
        placeholder takes its place; the kept peak heights are dropped so a
        Bins change on an idle page cannot build it again.  Idempotent;
        never raises."""
        self._release_plot("_detect_plot_widget",
                           ("_detect_thresh_line", "_detect_bar_item"),
                           placeholder_attr="_detect_plot_placeholder")
        self._detect_last_edt_values = []

    def release_add_plot(self) -> None:
        """Let the Add in-box histogram go at the end of an Add session,
        the same way as ``release_plots`` (an Add box that measures itself
        builds a fresh one).  A placeholder takes its place in the layout,
        as before the first box.  Idempotent; never raises."""
        self._release_plot("_add_plot_widget",
                           ("_add_thresh_line", "_add_bar_item"),
                           placeholder_attr="_add_plot_placeholder")

    def _release_plot(self, pw_attr, extra, placeholder_attr=None) -> None:
        pw = getattr(self, pw_attr, None)
        if pw is None:
            return
        setattr(self, pw_attr, None)
        for a in extra:
            setattr(self, a, None)
        try:
            from qtpy import sip as _sip

            if _sip.isdeleted(pw):
                return
        except Exception:
            pass
        try:
            if placeholder_attr is not None and getattr(
                    self, placeholder_attr, None) is None:
                parent = pw.parentWidget()
                lay = parent.layout() if parent is not None else None
                idx = lay.indexOf(pw) if lay is not None else -1
                if idx >= 0:
                    shown = pw.isVisibleTo(parent)
                    # parented from the start, and shown only once it is
                    # in the layout: never a top-level window, not even
                    # for a moment
                    ph = QWidget(parent)
                    ph.setFixedHeight(100)
                    ph.hide()
                    lay.removeWidget(pw)
                    lay.insertWidget(idx, ph)
                    ph.setVisible(shown)
                    setattr(self, placeholder_attr, ph)
            pw.hide()
            pw.setParent(None)
            pw.deleteLater()
        except Exception as exc:  # noqa: BLE001 - teardown never raises
            print(f"[ADV-DOCK] plot release {pw_attr}: {exc!r}")

    # --- Add histogram (mirrors the detect-seeds plot, separate widget) ---

    def _ensure_add_plot(self) -> None:
        """Lazily create the pyqtgraph in-box intensity histogram."""
        if self._add_plot_widget is not None:
            if self._add_plot_placeholder is None:
                return
            self._add_plot_widget = None
        if self._add_plot_placeholder is None:
            # released for a panel being torn down: nowhere to put a plot
            # (built anyway it would be parentless, a window of its own)
            return
        try:
            import pyqtgraph as pg
        except ImportError:
            return

        pw = pg.PlotWidget()
        pw.setBackground("#1e1e2e")
        pw.setFixedHeight(110)
        pw.getPlotItem().hideButtons()
        pw.setMouseEnabled(x=False, y=False)
        axis_style = {"color": "#aaa", "font-size": "9px"}
        pw.setLabel("bottom", "Intensity", **axis_style)
        pw.setLabel("left", "Count", **axis_style)
        pw.getAxis("bottom").setTextPen("#aaa")
        pw.getAxis("left").setTextPen("#aaa")
        line = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen("r", width=2, style=Qt.DashLine),
        )
        pw.addItem(line)
        line.sigPositionChanged.connect(self._on_add_line_moved)
        placeholder = self._add_plot_placeholder
        if placeholder is not None:
            parent_layout = placeholder.parentWidget().layout()
            if parent_layout is not None:
                idx = parent_layout.indexOf(placeholder)
                parent_layout.removeWidget(placeholder)
                placeholder.deleteLater()
                parent_layout.insertWidget(idx, pw)
            self._add_plot_placeholder = None
        self._add_plot_widget = pw
        self._add_bar_item = None
        self._add_thresh_line = line
        # a threshold set while there was no plot lives in the box: the
        # new line starts there (hidden while the threshold is pending)
        sb = getattr(self, "add_threshold_sb", None)
        if sb is None or getattr(self, "_add_threshold_is_pending", False):
            line.setVisible(False)
        else:
            line.blockSignals(True)
            line.setValue(float(sb.value()))
            line.blockSignals(False)

    def _on_add_line_moved(self, line) -> None:
        v = float(line.value())
        # the box follows the line under blockSignals and never round-trips
        # its 3-dp text into the value
        self._sync_add_threshold_box(v)
        cb = getattr(self, "_on_add_threshold_cb", None)
        if cb is not None:
            try:
                cb(v)
            except Exception:
                pass

    def _show_add_threshold_pending(self) -> None:
        """The threshold is not known yet (final fix pass B4): the box shows
        ``ADD_THRESHOLD_PENDING_TEXT`` (its special value text, at its
        minimum) and the line hides, until a value is pushed."""
        self._add_threshold_is_pending = True
        sb = getattr(self, "add_threshold_sb", None)
        if sb is not None:
            sb.blockSignals(True)
            try:
                sb.setSpecialValueText(ADD_THRESHOLD_PENDING_TEXT)
                sb.setValue(sb.minimum())
            finally:
                sb.blockSignals(False)
        line = getattr(self, "_add_thresh_line", None)
        if line is not None:
            try:
                line.setVisible(False)
            except Exception:
                pass

    def _sync_add_threshold_box(self, value: float) -> None:
        sb = getattr(self, "add_threshold_sb", None)
        if sb is None:
            return
        self._add_threshold_is_pending = False
        line = getattr(self, "_add_thresh_line", None)
        if line is not None:
            try:
                line.setVisible(True)
            except Exception:
                pass
        sb.blockSignals(True)
        try:
            sb.setSpecialValueText("")
            if value < sb.minimum() or value > sb.maximum():
                sb.setRange(min(sb.minimum(), value), max(sb.maximum(), value))
            sb.setValue(float(value))
        finally:
            sb.blockSignals(False)

    def set_add_nuclei_threshold(self, value) -> None:
        """Show the nuclei-source threshold (Auto) without firing."""
        sb = getattr(self, "add_nuclei_threshold_sb", None)
        if sb is None or value is None:
            return
        sb.blockSignals(True)
        try:
            sb.setValue(float(value))
        finally:
            sb.blockSignals(False)

    def set_add_source_note(self, text: str) -> None:
        """A run's refusal sentence, KEPT in the Markers help slot:
        ``_apply_add_visibility`` prefers it over the ordinary help
        while the run source is picked, so the next refresh cannot
        overwrite the reason S1 points at (B2 fix pass U-H2).  ``""``
        clears it; so does a source change."""
        self._add_source_note = str(text or "")
        self._apply_add_visibility()

    def _update_add_histogram(self, data: dict) -> None:
        """Redraw the in-box intensity histogram.

        ``data`` is ``{counts, edges}`` -- bins the box-info worker measured
        (Add rebuild B4.2; empty = pending, the bars cleared) -- or the raw
        ``{values, n_bins}`` form, binned here."""
        self._ensure_add_plot()
        if self._add_plot_widget is None:
            return
        try:
            import pyqtgraph as pg
        except ImportError:
            return
        import numpy as _np

        pw = self._add_plot_widget
        if self._add_bar_item is not None:
            pw.removeItem(self._add_bar_item)
            self._add_bar_item = None
        if "counts" in data:
            counts = _np.asarray(data.get("counts", []), dtype=float)
            edges = _np.asarray(data.get("edges", []), dtype=float)
            self._add_last_values = []
            if counts.size == 0 or edges.size != counts.size + 1:
                return
        else:
            values = _np.asarray(data.get("values", []), dtype=float)
            self._add_last_values = list(values)
            n_bins = int(data.get("n_bins", 32))
            if values.size == 0:
                return
            counts, edges = _np.histogram(values, bins=n_bins)
        widths = _np.diff(edges)
        bar = pg.BarGraphItem(
            x=edges[:-1], height=counts, width=widths * 0.9,
            brush="#4c78a8", pen=None,
        )
        pw.addItem(bar)
        self._add_bar_item = bar
        pw.setXRange(float(edges[0]), float(edges[-1]), padding=0.05)
        # the typed box spans the histogram's value range (0-1 once Local
        # contrast is on; the raw range otherwise)
        sb = getattr(self, "add_threshold_sb", None)
        if sb is not None:
            sb.blockSignals(True)
            try:
                lo, hi = float(edges[0]), float(edges[-1])
                pending = getattr(self, "_add_threshold_is_pending", False)
                if pending:
                    # nothing to keep in range: the new space's own range,
                    # the box back at its minimum = the pending text (B4)
                    sb.setRange(lo, hi)
                    sb.setValue(sb.minimum())
                else:
                    cur = float(sb.value())
                    sb.setRange(min(lo, cur), max(hi, cur))
                span = max(hi - lo, 1e-6)
                sb.setSingleStep(span / 100.0)
            finally:
                sb.blockSignals(False)

    def _set_add_threshold(self, value: float) -> None:
        """Move the threshold line without firing the callback."""
        # never BUILDS the plot (the box is typed into while Add is idle
        # too): only a box's histogram does (``_update_add_histogram``), and
        # the new line starts at the box's value
        if self._add_thresh_line is not None:
            self._add_thresh_line.blockSignals(True)
            self._add_thresh_line.setValue(value)
            self._add_thresh_line.blockSignals(False)
        self._sync_add_threshold_box(float(value))

    # --- Smooth Label (3D Curation) -------------------------------------------

    def _build_refine_section(self) -> QWidget:
        """Build the Refine page for 3D curation mode.

        Seven methods on the shared S1-S6 skeleton.  What Brick B1 changed,
        in the order the page reads:

        * a collapsed **description** above S1, the Reshape pattern;
        * **S2** is a read-only VALUE ROW -- the object id and its voxel
          count as text, never a dropdown, because the tool follows the 3D
          selection and a combo would invite a click that does nothing;
        * **S3** keeps the combo and the 7-page stack, and gains the ONE
          shared channel + prepare card every image-reading method uses;
        * **S4 is PINNED** between the scroll and the footer, and carries
          Preview, Stop, the busy bar and the stage line, with all three
          slots reserved so its height is the same in every state;
        * **S5** is new: the result card, the chain breadcrumb, the empty-
          object warning and the walls tick;
        * **S6** is unchanged in mechanism -- the secondary is keyed on
          "is a result staged" and says Cancel or Discard.
        """
        page, layout, footer, scroll = self._page_with_footer()
        content = scroll.widget()

        _hdr = QHBoxLayout()
        _hdr_label = QLabel("Refine one object")
        _hdr_label.setStyleSheet(SECTION_HEADER_STYLE)
        _hdr.addWidget(_hdr_label)
        _hdr.addStretch(1)
        _hdr.addWidget(_help_button(
            content,
            guide_path=str(_GUIDES_DIR / "annotator_3d_refine.md"),
            title="Refining the shape of one object -- 3D Annotator",
            tooltip="Help: Refine",
        ))
        layout.addLayout(_hdr)

        # -- the collapsed description, the Reshape pattern ----
        _about_row = QHBoxLayout()
        _about_row.setContentsMargins(0, 0, 0, 0)
        _about_row.setSpacing(4)
        self.refine_about_btn = QPushButton(_CHEVRON_CLOSED, content)
        self.refine_about_btn.setCheckable(True)
        self.refine_about_btn.setChecked(False)
        self.refine_about_btn.setStyleSheet(PANEL_CHEVRON_STYLE)
        self.refine_about_btn.setToolTip(
            "What this tool does, in plain words.")
        self.refine_about_btn.setFixedWidth(_about_btn_width(self.refine_about_btn))
        _about_row.addWidget(self.refine_about_btn, 0, Qt.AlignTop)
        self._refine_about_line = _ClickableLabel(
            self.refine_about_btn.toggle, content)
        self._refine_about_line.setText(REFINE_ABOUT_LINE)
        self._refine_about_line.setStyleSheet(
            PANEL_HELP_STYLE + _ClickableLabel.HOVER_STYLE)
        self._refine_about_line.setWordWrap(True)
        self._refine_about_line.setToolTip(
            "What this tool does, in plain words.")
        _about_row.addWidget(self._refine_about_line, 1)
        layout.addLayout(_about_row)

        self._refine_about_body = QLabel(content)
        self._refine_about_body.setTextFormat(Qt.RichText)
        self._refine_about_body.setText(
            REFINE_ABOUT_BODY.replace("\n\n", "<br><br>"))
        self._refine_about_body.setStyleSheet(PANEL_HELP_STYLE)
        self._refine_about_body.setWordWrap(True)
        self._refine_about_body.setVisible(False)
        layout.addWidget(self._refine_about_body)
        self.refine_about_btn.toggled.connect(self._on_refine_about_toggled)

        # ==================================================================
        # S1 STATUS
        # ==================================================================
        self.smooth_status_label = QLabel(
            "Select an object, then pick a method."
        )
        self.smooth_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.smooth_status_label.setWordWrap(True)
        self._refine_s1_container = self._status_row(
            "_refine_status_dot", self.smooth_status_label
        )
        layout.addWidget(self._refine_s1_container)

        # ==================================================================
        # S2 OBJECT -- a card with a READ-ONLY VALUE ROW.  Refine acts on
        # whatever object is selected and there is no Start, so the slot
        # says which object that is and what size it is, in text.  A combo
        # here would invite a click that does nothing.
        # ==================================================================
        self._refine_target_card = QWidget()
        _s2 = QVBoxLayout(self._refine_target_card)
        make_panel_card(self._refine_target_card)
        _s2.addWidget(panel_card_header("Object", self._refine_target_card))

        _obj_row = QHBoxLayout()
        _obj_row.setSpacing(6)
        _obj_row.addWidget(panel_form_label("Selected", parent=self))
        self.refine_object_value = QLabel("-")
        self.refine_object_value.setStyleSheet(PANEL_HELP_STYLE)
        self.refine_object_value.setWordWrap(True)
        _obj_row.addWidget(self.refine_object_value, 1)
        _s2.addLayout(_obj_row)

        self._refine_target_hint = QLabel(
            "Follows the object selected in the 3D view or the Label "
            "Navigator."
        )
        self._refine_target_hint.setWordWrap(True)
        self._refine_target_hint.setStyleSheet(_HELPER_LINE_STYLE)
        _s2.addWidget(self._refine_target_hint)
        layout.addWidget(self._refine_target_card)

        # ==================================================================
        # S3 METHOD -- the combo, the description, the 7-page stack, and
        # the ONE shared channel card underneath them.
        # ==================================================================
        self._refine_method_card = QWidget()
        _s3 = QVBoxLayout(self._refine_method_card)
        make_panel_card(self._refine_method_card)
        _s3.addWidget(panel_card_header("Method", self._refine_method_card))

        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_row.addWidget(panel_form_label("Method", parent=self))
        self.refine_method_combo = QComboBox()
        self.refine_method_combo.setStyleSheet(PANEL_INPUT_STYLE)
        # THE WIRE FORMAT.  Filled from the module constant so the combo,
        # the stack, the descriptions and the annotator's dispatch cannot
        # drift apart.
        self.refine_method_combo.addItems(
            [text for _i, text, _k in REFINE_METHOD_KEYS])
        self._apply_item_tooltips("refine_method_combo")
        self.refine_method_combo.setToolTip(
            "Which change to make. Hover over a method in the open list to "
            "read what it does.")
        method_row.addWidget(self.refine_method_combo, 1)
        _s3.addLayout(method_row)

        self._refine_method_desc_label = QLabel(
            self._REFINE_METHOD_DESCRIPTIONS[0])
        self._refine_method_desc_label.setStyleSheet(_HELPER_LINE_STYLE)
        self._refine_method_desc_label.setWordWrap(True)
        _s3.addWidget(self._refine_method_desc_label)

        # --- Stacked parameter panels, ONE PER ENTRY in the constant ------
        self._refine_params_stack = _CurrentPageStack()
        for _idx, _text, _key in REFINE_METHOD_KEYS:
            self._refine_params_stack.addWidget(
                getattr(self, "_build_refine_page_" + _key)()
            )
        _s3.addWidget(self._refine_params_stack)

        # --- the ONE channel card ------------------------------
        self._refine_channel_container = QWidget()
        _ch = QVBoxLayout(self._refine_channel_container)
        _ch.setContentsMargins(0, 0, 0, 0)
        _ch.setSpacing(2)

        _ch_row = QHBoxLayout()
        _ch_row.setSpacing(6)
        _ch_row.addWidget(panel_form_label("Channel", parent=self))
        self.refine_channel_combo = QComboBox()
        self.refine_channel_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_channel_combo.setToolTip(
            HELP_LINES["refine_channel_combo"])
        _ch_row.addWidget(self.refine_channel_combo, 1)
        _ch.addLayout(_ch_row)
        _ch.addWidget(self._helper_line("refine_channel_combo"))

        # "Prepare the channel first" -- the SAME five ticks, the same
        # labels, the same order sentence, the same `chain_from_ticks`
        #.  Which chain a method gets is the CONSUMER's rule,
        # not a second ticklist: a method that builds a wall elevation
        # takes it with `drop_clahe=True`, so "Local contrast" is HIDDEN
        # on those methods rather than silently ignored.
        self.refine_prep_btn = self._subsection_chevron(
            "Prepare the channel first: none",
            "Clean the channel up before the tool reads it.",
        )
        self.refine_prep_btn.setProperty("_epicycle_elide_title", True)
        self.refine_prep_btn.setStyleSheet(PANEL_CHIP_STYLE)
        _ch.addWidget(self.refine_prep_btn)

        self._refine_prep_container = QWidget()
        _pc = QVBoxLayout(self._refine_prep_container)
        _pc.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _pc.setSpacing(1)
        self._refine_prep_rows = {}
        for _key in ("clahe", "zcorrect", "bgsub", "lift", "denoise"):
            _attr = "refine_prep_{}_cb".format(_key)
            _cb = QCheckBox(PREP_TICK_LABELS[_key])
            _cb.setChecked(_key == "clahe")
            _cb.setToolTip(HELP_LINES[_attr])
            setattr(self, _attr, _cb)
            _line = self._helper_line(_attr)
            _pc.addWidget(_cb)
            _pc.addWidget(_line)
            self._refine_prep_rows[_key] = (_cb, _line)
            _cb.toggled.connect(self._refresh_refine_prep_chip)
        self._refine_prep_no_clahe_line = QLabel(REFINE_NO_CLAHE_SENTENCE)
        self._refine_prep_no_clahe_line.setWordWrap(True)
        self._refine_prep_no_clahe_line.setStyleSheet(_HELPER_LINE_STYLE)
        _pc.addWidget(self._refine_prep_no_clahe_line)
        self._refine_prep_order = QLabel(_PREP_ORDER_SENTENCE)
        self._refine_prep_order.setWordWrap(True)
        self._refine_prep_order.setStyleSheet(_HELPER_LINE_STYLE)
        _pc.addWidget(self._refine_prep_order)
        self._refine_prep_container.setVisible(False)
        _ch.addWidget(self._refine_prep_container)

        # The line that replaces the whole card when the wall kind reads
        # the reconstruction rather than a channel.
        self._refine_no_channel_line = QLabel(
            "Uses the reconstruction's probability map - no channel needed."
        )
        self._refine_no_channel_line.setWordWrap(True)
        self._refine_no_channel_line.setStyleSheet(_HELPER_LINE_STYLE)
        self._refine_no_channel_line.setVisible(False)
        _s3.addWidget(self._refine_channel_container)
        _s3.addWidget(self._refine_no_channel_line)
        self._refine_wall_container = self._build_refine_wall_row()
        _s3.addWidget(self._refine_wall_container)
        # the contour's rows below Mode, under the shared block (UX-M5)
        _s3.addWidget(self._refine_contour_lower)
        layout.addWidget(self._refine_method_card)

        self.refine_prep_btn.toggled.connect(self._on_refine_prep_toggled)

        # ==================================================================
        # ADVANCED -- at the BOTTOM of the scroll content, per the skeleton.
        # ==================================================================
        self.refine_advanced_btn = self._subsection_chevron(
            "Advanced", "Knobs you rarely need.")
        layout.addWidget(self.refine_advanced_btn)
        self._refine_advanced_container = QWidget()
        _adv = QVBoxLayout(self._refine_advanced_container)
        _adv.setContentsMargins(_SUB_INDENT, 0, 0, 0)
        _adv.setSpacing(2)

        self._refine_settle_advanced = QWidget()
        _sadv = QVBoxLayout(self._refine_settle_advanced)
        _sadv.setContentsMargins(0, 0, 0, 0)
        _sadv.setSpacing(2)
        _sadv.addWidget(panel_card_header(
            "Settle", self._refine_settle_advanced))

        self.refine_settle_clamp_sb = QDoubleSpinBox()
        self.refine_settle_clamp_sb.setRange(0.05, 0.95)
        self.refine_settle_clamp_sb.setSingleStep(0.05)
        self.refine_settle_clamp_sb.setDecimals(2)
        self.refine_settle_clamp_sb.setValue(0.50)
        self.refine_settle_clamp_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_settle_clamp_sb.setMinimumWidth(
            self._readable_spin_width(self.refine_settle_clamp_sb))
        self.refine_settle_clamp_sb.setToolTip(
            HELP_LINES["refine_settle_clamp_sb"])
        _sadv.addLayout(self._refine_row(
            "Hold the outline", self.refine_settle_clamp_sb, advanced=True))
        _sadv.addWidget(self._helper_line("refine_settle_clamp_sb"))

        self.refine_settle_compactness_sb = QDoubleSpinBox()
        self.refine_settle_compactness_sb.setRange(0.0, 1.0)
        self.refine_settle_compactness_sb.setSingleStep(0.05)
        self.refine_settle_compactness_sb.setDecimals(2)
        self.refine_settle_compactness_sb.setValue(0.0)
        self.refine_settle_compactness_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_settle_compactness_sb.setMinimumWidth(
            self._readable_spin_width(self.refine_settle_compactness_sb))
        self.refine_settle_compactness_sb.setToolTip(
            HELP_LINES["refine_settle_compactness_sb"])
        _sadv.addLayout(self._refine_row(
            "Balance", self.refine_settle_compactness_sb, advanced=True))
        _sadv.addWidget(self._helper_line("refine_settle_compactness_sb"))
        _adv.addWidget(self._refine_settle_advanced)

        # The contour's Advanced card: three knobs that were
        # hard-coded or dead, and the 2D rebuild's blur.
        self._refine_contour_advanced = QWidget()
        _cadv = QVBoxLayout(self._refine_contour_advanced)
        _cadv.setContentsMargins(0, 0, 0, 0)
        _cadv.setSpacing(2)
        _cadv.addWidget(panel_card_header(
            "Active contour", self._refine_contour_advanced))
        self._refine_contour_snap_adv = QWidget()
        _csa = QVBoxLayout(self._refine_contour_snap_adv)
        _csa.setContentsMargins(0, 0, 0, 0)
        _csa.setSpacing(2)
        self.refine_contour_gate_sb = QDoubleSpinBox()
        self.refine_contour_gate_sb.setRange(0.0, 1.0)
        self.refine_contour_gate_sb.setSingleStep(0.05)
        self.refine_contour_gate_sb.setDecimals(2)
        self.refine_contour_gate_sb.setValue(0.30)
        self.refine_contour_gate_sb.setToolTip(
            HELP_LINES["refine_contour_gate_sb"])
        _csa.addLayout(self._refine_row(
            "Edge gate", self.refine_contour_gate_sb, advanced=True))
        _csa.addWidget(self._helper_line("refine_contour_gate_sb"))
        self.refine_contour_blur_sb = QDoubleSpinBox()
        self.refine_contour_blur_sb.setRange(0.5, 10.0)
        self.refine_contour_blur_sb.setSingleStep(0.5)
        self.refine_contour_blur_sb.setDecimals(1)
        self.refine_contour_blur_sb.setValue(2.0)
        self.refine_contour_blur_sb.setSuffix(" um")
        self.refine_contour_blur_sb.setToolTip(
            HELP_LINES["refine_contour_blur_sb"])
        _csa.addLayout(self._refine_row(
            "Blur", self.refine_contour_blur_sb, advanced=True))
        _csa.addWidget(self._helper_line("refine_contour_blur_sb"))
        self.refine_contour_sharpness_sb = QSpinBox()
        self.refine_contour_sharpness_sb.setRange(1, 1000)
        self.refine_contour_sharpness_sb.setSingleStep(10)
        self.refine_contour_sharpness_sb.setValue(100)
        self.refine_contour_sharpness_sb.setToolTip(
            HELP_LINES["refine_contour_sharpness_sb"])
        _csa.addLayout(self._refine_row(
            "Edge sharpness", self.refine_contour_sharpness_sb,
            advanced=True))
        _csa.addWidget(self._helper_line("refine_contour_sharpness_sb"))
        _cadv.addWidget(self._refine_contour_snap_adv)
        self._refine_contour_rebuild_adv = QWidget()
        _cra = QVBoxLayout(self._refine_contour_rebuild_adv)
        _cra.setContentsMargins(0, 0, 0, 0)
        _cra.setSpacing(2)
        # In MICRONS, like every other contour number,
        # with the voxel equivalent through the chosen plane's
        # depth as a read-out beside it.
        self.refine_contour_rebuild_um_sb = QDoubleSpinBox()
        self.refine_contour_rebuild_um_sb.setRange(0.0, 15.0)
        self.refine_contour_rebuild_um_sb.setSingleStep(0.5)
        self.refine_contour_rebuild_um_sb.setDecimals(1)
        self.refine_contour_rebuild_um_sb.setValue(1.5)
        self.refine_contour_rebuild_um_sb.setSuffix(" um")
        self.refine_contour_rebuild_um_sb.setToolTip(
            HELP_LINES["refine_contour_rebuild_um_sb"])
        _cra.addLayout(self._refine_row(
            "Smooth the join", self.refine_contour_rebuild_um_sb,
            advanced=True))
        _cra.addWidget(self._helper_line("refine_contour_rebuild_um_sb"))
        self.refine_contour_rebuild_readout = QLabel("")
        self.refine_contour_rebuild_readout.setWordWrap(True)
        self.refine_contour_rebuild_readout.setStyleSheet(_HELPER_LINE_STYLE)
        _cra.addWidget(self.refine_contour_rebuild_readout)
        self.refine_contour_rebuild_um_sb.valueChanged.connect(
            self._refresh_refine_readouts)
        self._refine_contour_rebuild_adv.setVisible(False)
        _cadv.addWidget(self._refine_contour_rebuild_adv)
        self._refine_contour_advanced.setVisible(False)
        _adv.addWidget(self._refine_contour_advanced)
        for _sb in (self.refine_contour_gate_sb,
                    self.refine_contour_sharpness_sb):
            _sb.valueChanged.connect(self._refresh_refine_push_line)

        # The probability overlay pair keeps its home, at the bottom of
        # Advanced: it is a VIEWING aid, not a parameter of any method.
        prob_row = QHBoxLayout()
        prob_row.setSpacing(4)
        self.refine_prob_overlay_cb = QCheckBox("Show the probability map")
        self.refine_prob_overlay_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.refine_prob_overlay_cb.setToolTip(
            HELP_LINES["refine_prob_overlay_cb"])
        prob_row.addWidget(self.refine_prob_overlay_cb)
        self.refine_prob_cmap_combo = QComboBox()
        self.refine_prob_cmap_combo.addItems([
            "viridis", "magma", "inferno", "plasma", "turbo",
            "hot", "cool", "gray",
        ])
        self.refine_prob_cmap_combo.setToolTip("Colours for that overlay.")
        self.refine_prob_cmap_combo.setMaximumWidth(90)
        prob_row.addWidget(self.refine_prob_cmap_combo)
        prob_row.addStretch(1)
        _adv.addLayout(prob_row)
        _adv.addWidget(self._helper_line("refine_prob_overlay_cb"))
        # The on-screen reason when the tick could not show anything.
        self.refine_prob_overlay_msg = QLabel("")
        self.refine_prob_overlay_msg.setWordWrap(True)
        self.refine_prob_overlay_msg.setStyleSheet(PANEL_WARN_STYLE)
        self.refine_prob_overlay_msg.setVisible(False)
        _adv.addWidget(self.refine_prob_overlay_msg)

        self._refine_advanced_container.setVisible(False)
        layout.addWidget(self._refine_advanced_container)
        self.refine_advanced_btn.toggled.connect(
            self._on_refine_advanced_toggled)

        layout.addStretch(1)

        self.refine_method_combo.currentIndexChanged.connect(
            self._on_refine_method_changed
        )

        # ==================================================================
        # S4 PREVIEW -- PINNED between the scroll area and the footer, the
        # Reshape recipe (`_split_s4_container`).  Preview is needed in
        # every state, and S3 on this page carries a 7-page stack plus a
        # channel card whose prepare block can be taller than the viewport,
        # so a scrolling Preview is a Preview the user cannot reach.
        #
        # Its height is asserted CONSTANT across every state, so all three
        # slots -- the bar and the stage line included -- are reserved at
        # construction and merely shown and hidden inside that box.
        # ==================================================================
        self._refine_s4_container = QWidget(page)
        _s4 = QVBoxLayout(self._refine_s4_container)
        _s4.setContentsMargins(0, 0, 0, 0)
        _s4.setSpacing(2)

        preview_row = QWidget()
        _pr = QHBoxLayout(preview_row)
        _pr.setContentsMargins(0, 0, 0, 0)
        _pr.setSpacing(4)
        self.smooth_preview_btn = QPushButton("Preview")
        self.smooth_preview_btn.setStyleSheet(PANEL_PRIMARY_BTN_STYLE)
        self.smooth_preview_btn.setToolTip(REFINE_PREVIEW_TIP)
        _pr.addWidget(self.smooth_preview_btn, 1)

        self.refine_cancel_preview_btn = QPushButton("Stop")
        self.refine_cancel_preview_btn.setStyleSheet(PANEL_BTN_STYLE)
        # `_on_refine_stopped` writes the SESSION snapshot back: the object
        # goes back to before the first step of a chain, not the last.
        self.refine_cancel_preview_btn.setToolTip(
            "Stops, and puts the object back as it was before the first step."
        )
        self.refine_cancel_preview_btn.setVisible(False)
        _pr.addWidget(self.refine_cancel_preview_btn)

        # Refine recipes: ONE menu button in the row that
        # is already there, so it costs 0 px of height in every state.  Qt's
        # own arrow (the menu-indicator rule), not a character in the text.
        # The menu is rebuilt on every `aboutToShow` from the annotator's
        # live state (`populate_refine_recipes_menu`).
        self.refine_recipes_menu = QMenu(self)
        self.refine_recipes_menu.setToolTipsVisible(True)
        self.refine_recipes_btn = QPushButton("Recipes")
        self.refine_recipes_btn.setStyleSheet(
            PANEL_BTN_STYLE + _RECIPES_MENU_INDICATOR_STYLE)
        self.refine_recipes_btn.setToolTip(REFINE_RECIPES_TIP)
        self.refine_recipes_btn.setMenu(self.refine_recipes_menu)
        _pr.addWidget(self.refine_recipes_btn)
        self._refine_recipe_cbs = {}
        self.populate_refine_recipes_menu([(
            "info", "Recipes need a workspace \u2014 open this from Step 3.",
            False, None, "")])
        _s4.addWidget(preview_row)

        self._refine_busy_container = QWidget()
        _busy = QHBoxLayout(self._refine_busy_container)
        _busy.setContentsMargins(0, 0, 0, 0)
        _busy.setSpacing(4)
        self.refine_busy_bar = QProgressBar()
        self.refine_busy_bar.setRange(0, 0)          # marquee until sized
        self.refine_busy_bar.setTextVisible(False)
        self.refine_busy_bar.setFixedHeight(4)
        self.refine_busy_bar.setStyleSheet(_PANEL_BUSY_BAR_STYLE)
        _busy.addWidget(self.refine_busy_bar, 1)
        self._refine_busy_container.setVisible(False)
        _s4.addWidget(self._refine_busy_container)

        self.refine_stage_label = QLabel("")
        self.refine_stage_label.setStyleSheet(_HELPER_LINE_STYLE)
        # ONE line, never two: this strip's height is asserted constant, and
        # a wrapping stage line is the one thing in it that can grow.  The
        # whole sentence rides in the tooltip.
        self.refine_stage_label.setWordWrap(False)
        self.refine_stage_label.setVisible(False)
        # a stage line may now carry "3/3 · Plane
        # 4 of 13 ..." or "'<40 characters>' stopped at 2/3 · ...", so it
        # must never ask for width: horizontally IGNORED, right-elided to
        # the width it is given, the whole sentence in the tooltip.
        self.refine_stage_label.setSizePolicy(
            QSizePolicy.Ignored,
            self.refine_stage_label.sizePolicy().verticalPolicy())
        self._refine_stage_full_text = ""
        self._refine_stage_elider = _ElideOnResize(
            self._apply_refine_stage_elision, self.refine_stage_label)
        self.refine_stage_label.installEventFilter(self._refine_stage_elider)
        _s4.addWidget(self.refine_stage_label)

        # Reserve ALL THREE slots, so the strip does not grow when a
        # preview starts (Reshape reserves only the bar, because it has no
        # stage line).  The height comes from the widgets' OWN metrics, so
        # it scales with the machine's font rather than assuming this one.
        # Measured from the widgets themselves, with all three slots
        # SHOWING, and then FIXED -- so the bar and the stage line appear
        # inside reserved space rather than adding to it.
        self._refine_busy_container.setVisible(True)
        self.refine_stage_label.setText("X")
        self.refine_stage_label.setVisible(True)
        _reserved = max(
            self._refine_s4_container.sizeHint().height(),
            preview_row.sizeHint().height() + 2 * _s4.spacing()
            + self.refine_busy_bar.height()
            + self.refine_stage_label.sizeHint().height(),
        )
        self._refine_busy_container.setVisible(False)
        self.refine_stage_label.setText("")
        self.refine_stage_label.setVisible(False)
        self._refine_s4_container.setFixedHeight(int(_reserved))

        page.layout().insertWidget(
            page.layout().indexOf(footer), self._refine_s4_container, 0
        )

        # ==================================================================
        # S5 RESULT -- ABOVE the footer, hidden until something is staged.
        # ==================================================================
        self._refine_result_container = QWidget()
        self._refine_result_container.setVisible(False)
        _s5 = QVBoxLayout(self._refine_result_container)
        make_panel_card(self._refine_result_container)

        _s5_head = QHBoxLayout()
        _s5_head.setContentsMargins(0, 0, 0, 0)
        _s5_head.setSpacing(6)
        _s5_head.addWidget(
            panel_card_header("Result", self._refine_result_container))
        _s5_head.addStretch(1)
        # The walls tick rides the RESULT card's TITLE ROW, for the reason
        # UX-round 5 measured on Reshape: a row of its own is 15 px and the
        # header row is already there.
        self.refine_show_walls_cb = QCheckBox(
            "Show what the tool sees", self._refine_result_container)
        self.refine_show_walls_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.refine_show_walls_cb.setEnabled(False)
        self.refine_show_walls_cb.setToolTip(WALLS_TOGGLE_DISABLED)
        _s5_head.addWidget(self.refine_show_walls_cb, 0)
        _s5.addLayout(_s5_head)

        self.refine_walls_legend = QLabel("", self._refine_result_container)
        self.refine_walls_legend.setWordWrap(True)
        self.refine_walls_legend.setStyleSheet(PANEL_HELP_STYLE)
        self.refine_walls_legend.setVisible(False)
        _s5.addWidget(self.refine_walls_legend)

        # The chain, in one line and one word: "steps", everywhere.
        self.refine_breadcrumb = QLabel("", self._refine_result_container)
        self.refine_breadcrumb.setWordWrap(True)
        self.refine_breadcrumb.setStyleSheet(PANEL_HELP_STYLE)
        self.refine_breadcrumb.setVisible(False)
        _s5.addWidget(self.refine_breadcrumb)

        # What changed, naming its METHOD -- so the line is self-describing
        # even when the combo above it has since been changed.
        self.refine_result_note = QLabel("", self._refine_result_container)
        self.refine_result_note.setWordWrap(True)
        self.refine_result_note.setStyleSheet(PANEL_HELP_STYLE)
        self.refine_result_note.setVisible(False)
        _s5.addWidget(self.refine_result_note)

        # The red block: "this removes the whole object", the spacing
        # caveat, an engine refusal.
        self.refine_result_warning = QLabel("", self._refine_result_container)
        self.refine_result_warning.setWordWrap(True)
        self.refine_result_warning.setStyleSheet(PANEL_ERROR_STYLE)
        self.refine_result_warning.setVisible(False)
        _s5.addWidget(self.refine_result_warning)

        # Into the SCROLL CONTENT, directly above Advanced -- the Reshape
        # arrangement exactly (`_split_result_container`, :3060).  Only S4
        # is pinned; a page is scroll -> optional pinned strip -> footer,
        # and a second pinned thing would break the height assertion the
        # strip's constancy rests on.
        layout.insertWidget(
            layout.indexOf(self.refine_advanced_btn),
            self._refine_result_container,
        )

        # ==================================================================
        # S6 FOOTER -- two buttons and one line, and nothing else.
        # ==================================================================
        _ft = QVBoxLayout(footer)
        _ft.setContentsMargins(4, 2, 4, 2)
        _ft.setSpacing(2)

        btn_row = QHBoxLayout()
        self.smooth_accept_btn = QPushButton("Accept")
        self.smooth_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.smooth_accept_btn.setEnabled(False)
        # Two words, one meaning each: "Cancel" while nothing is staged,
        # "Discard" once a refined shape is waiting for a decision.
        self.smooth_reject_btn = QPushButton("Cancel")
        self.smooth_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)
        self.smooth_reject_btn.setEnabled(False)
        self.smooth_reject_btn.setToolTip(
            "Put the object back exactly as it was, however many refine "
            "steps you chained onto it."
        )
        # `apply_smooth_to_volume` / `apply_settle_to_volume` write only
        # into this object's old voxels and empty ones.
        self.smooth_accept_btn.setToolTip(
            "Keep the refined shape. It changes only this object and empty "
            "space, never a neighbour."
        )
        btn_row.addWidget(self.smooth_accept_btn)
        btn_row.addWidget(self.smooth_reject_btn)
        _ft.addLayout(btn_row)

        self._refine_footer_summary = self._footer_summary_label()
        _ft.addWidget(self._refine_footer_summary)
        self._refresh_refine_footer_summary()

        self._register_page(
            "Refine", page, scroll, content, footer,
            status=self.smooth_status_label, preview=self.smooth_preview_btn,
        )
        # One pass so idle agrees with the combo's index 0.
        self._apply_refine_method_visibility()
        return page

    # --- the seven parameter pages -----------------------------------------
    #
    # One method per page, named for its ENGINE KEY, so
    # `_build_refine_section`'s loop over `REFINE_METHOD_KEYS` is the only
    # place the order lives.

    def _refine_row(self, label, field, *, advanced=False):
        """One parameter row: the fixed label column, then the field.

        NOT a `QFormLayout`: that asks for the label's full width plus the
        field's, and with the horizontal scrollbar off the overflow is
        clipped rather than scrolled.  Measured at the 340 px dock, the
        Smooth page wanted 478 px of a 329 px viewport that way.
        """
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(panel_form_label(
            str(label),
            PANEL_ADV_LABEL_COLUMN if advanced else PANEL_LABEL_COLUMN,
            parent=self))
        try:
            field.setStyleSheet(PANEL_INPUT_STYLE)
            if isinstance(field, (QSpinBox, QDoubleSpinBox)):
                field.setMinimumWidth(self._readable_spin_width(field))
        except Exception:
            pass
        row.addWidget(field, 1)
        return row

    def _build_refine_page_sdf(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self.smooth_standalone_sigma_z_sb = QDoubleSpinBox()
        self.smooth_standalone_sigma_z_sb.setRange(0.1, 10.0)
        self.smooth_standalone_sigma_z_sb.setSingleStep(0.5)
        self.smooth_standalone_sigma_z_sb.setDecimals(1)
        self.smooth_standalone_sigma_z_sb.setValue(3.0)
        self.smooth_standalone_sigma_z_sb.setToolTip(
            HELP_LINES["smooth_standalone_sigma_z_sb"])
        lay.addLayout(self._refine_row(
            "Blur through depth", self.smooth_standalone_sigma_z_sb))
        lay.addWidget(self._helper_line("smooth_standalone_sigma_z_sb"))

        self.smooth_standalone_sigma_xy_sb = QDoubleSpinBox()
        self.smooth_standalone_sigma_xy_sb.setRange(0.1, 10.0)
        self.smooth_standalone_sigma_xy_sb.setSingleStep(0.1)
        self.smooth_standalone_sigma_xy_sb.setDecimals(1)
        self.smooth_standalone_sigma_xy_sb.setValue(1.5)
        self.smooth_standalone_sigma_xy_sb.setToolTip(
            HELP_LINES["smooth_standalone_sigma_xy_sb"])
        lay.addLayout(self._refine_row(
            "Blur within a slice", self.smooth_standalone_sigma_xy_sb))
        lay.addWidget(self._helper_line("smooth_standalone_sigma_xy_sb"))

        # The live micron read-out: the number stays in voxels
        # -- a physical sigma would change every existing result -- and
        # what it means on THIS patch is said beside it.
        self.refine_sdf_um_label = QLabel("")
        self.refine_sdf_um_label.setWordWrap(True)
        self.refine_sdf_um_label.setStyleSheet(_HELPER_LINE_STYLE)
        lay.addWidget(self.refine_sdf_um_label)

        self.smooth_preserve_topology_cb = QCheckBox(
            "Keep the object in one piece")
        self.smooth_preserve_topology_cb.setStyleSheet(PANEL_CHECKBOX_STYLE)
        self.smooth_preserve_topology_cb.setChecked(True)
        # B3: the old tooltip said "leave it on unless the object is
        # already in pieces" -- but `smooth_label_sdf` keeps only the
        # LARGEST component afterwards whichever way this is set, so that
        # advice promised something no setting delivers.
        self.smooth_preserve_topology_cb.setToolTip(
            HELP_LINES["smooth_preserve_topology_cb"])
        lay.addWidget(self.smooth_preserve_topology_cb)
        lay.addWidget(self._helper_line("smooth_preserve_topology_cb"))
        for _sb in (self.smooth_standalone_sigma_z_sb,
                    self.smooth_standalone_sigma_xy_sb):
            _sb.valueChanged.connect(self._refresh_refine_readouts)
        return w

    def _build_refine_page_contour(self) -> QWidget:
        """Active contour -- one method, two modes.

        The tallest page the tab has, so the four trims are
        taken here: the mode's line describes the CHOSEN mode only
        (`ITEM_HELP`), Wall type and its rows leave in the brightness mode,
        Orientation's explanation is a tooltip, and the wall-type line is
        not repeated.  Every knob is read BY INDEX or by value through
        `get_refine_params()`.
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(panel_form_label("Mode", parent=self))
        self.refine_contour_mode_combo = QComboBox()
        self.refine_contour_mode_combo.addItems(
            list(REFINE_CONTOUR_MODE_LABELS))
        self.refine_contour_mode_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_contour_mode_combo.setToolTip(
            "Snap to edges moves the outline onto a visible wall. Inside vs "
            "outside brightness compares a band just inside the outline "
            "with a band just outside - for when the wall is not visible.")
        mode_row.addWidget(self.refine_contour_mode_combo, 1)
        lay.addLayout(mode_row)
        lay.addWidget(self._item_help_label("refine_contour_mode_combo"))

        # Everything below Mode is the LOWER block (B2 audit UX-M5): the
        # mockup puts the shared Channel -> Prepare -> Wall type rows ABOVE
        # Orientation, and those live under the stack -- so this block is
        # placed under them by `_build_refine_section`, and shown for the
        # contour only.
        self._refine_contour_lower = QWidget()
        _lo = QVBoxLayout(self._refine_contour_lower)
        _lo.setContentsMargins(0, 0, 0, 0)
        _lo.setSpacing(2)

        # Orientation -- trim 3: a four-item combo whose items explain
        # themselves, so the sentence is the TOOLTIP, not a grey line.
        self.refine_contour_axis_combo = QComboBox()
        self.refine_contour_axis_combo.addItems(list(REFINE_CONTOUR_AXES))
        self.refine_contour_axis_combo.setToolTip(
            "XY/XZ/YZ trace some slices in that plane and fill in the rest.")
        _lo.addLayout(self._refine_row(
            "Orientation", self.refine_contour_axis_combo))

        # The slice sampler -- only for a plane orientation.
        self._refine_contour_sampling = QWidget()
        _smp = QVBoxLayout(self._refine_contour_sampling)
        _smp.setContentsMargins(0, 0, 0, 0)
        _smp.setSpacing(2)
        self.refine_contour_slices_combo = QComboBox()
        self.refine_contour_slices_combo.addItems(
            list(REFINE_CONTOUR_SLICE_LABELS))
        self.refine_contour_slices_combo.setToolTip(
            "Which slices are traced: a share, a fixed number, or every "
            "slice. The rest are filled in.")
        _smp.addLayout(self._refine_row(
            "Slices", self.refine_contour_slices_combo))
        self._refine_contour_share_row = QWidget()
        _sh = QVBoxLayout(self._refine_contour_share_row)
        _sh.setContentsMargins(0, 0, 0, 0)
        self.refine_contour_share_sb = QSpinBox()
        self.refine_contour_share_sb.setRange(5, 100)
        self.refine_contour_share_sb.setSingleStep(5)
        self.refine_contour_share_sb.setValue(25)
        self.refine_contour_share_sb.setSuffix(" %")
        self.refine_contour_share_sb.setToolTip(
            "What share of the object's slices is traced. Never fewer than "
            "5 slices, never more than 60.")
        _sh.addLayout(self._refine_row("Share", self.refine_contour_share_sb))
        _smp.addWidget(self._refine_contour_share_row)
        self._refine_contour_nslices_row = QWidget()
        _nr = QVBoxLayout(self._refine_contour_nslices_row)
        _nr.setContentsMargins(0, 0, 0, 0)
        self.refine_contour_nslices_sb = QSpinBox()
        self.refine_contour_nslices_sb.setRange(3, 60)
        self.refine_contour_nslices_sb.setValue(12)
        self.refine_contour_nslices_sb.setToolTip(
            "How many slices are traced, spread evenly. At most 60.")
        _nr.addLayout(self._refine_row(
            "Slices traced", self.refine_contour_nslices_sb))
        self._refine_contour_nslices_row.setVisible(False)
        _smp.addWidget(self._refine_contour_nslices_row)
        _smp.addWidget(self._item_help_label("refine_contour_slices_combo"))
        self._refine_contour_sampling.setVisible(False)
        _lo.addWidget(self._refine_contour_sampling)

        # How far -- in MICRONS, both ways.
        self.refine_contour_limit_sb = QDoubleSpinBox()
        self.refine_contour_limit_sb.setRange(0.5, 30.0)
        self.refine_contour_limit_sb.setSingleStep(0.5)
        self.refine_contour_limit_sb.setDecimals(1)
        self.refine_contour_limit_sb.setValue(3.0)
        self.refine_contour_limit_sb.setSuffix(" um")
        self.refine_contour_limit_sb.setToolTip(
            "A hard limit in both directions: no point of the outline may "
            "end up further than this from where it started, out or in.")
        _lo.addLayout(self._refine_row(
            "Edge may move", self.refine_contour_limit_sb))
        # No grey line under it (B2 audit UX-M4, the rule  records):
        # the card sentence "Local contrast is not used..." was the next row
        # added to the tallest page, so the named candidate came off -- the
        # sentence is the knob's TOOLTIP.

        self.refine_contour_iterations_sb = QSpinBox()
        self.refine_contour_iterations_sb.setRange(10, 500)
        self.refine_contour_iterations_sb.setSingleStep(10)
        self.refine_contour_iterations_sb.setValue(100)
        self.refine_contour_iterations_sb.setToolTip(
            "How many steps the outline takes. More lets it travel further, "
            "up to the limit above.")
        _lo.addLayout(self._refine_row(
            "Steps", self.refine_contour_iterations_sb))

        # Push outward -- Snap to edges only, with its LIVE line.
        self._refine_contour_push_box = QWidget()
        _pb = QVBoxLayout(self._refine_contour_push_box)
        _pb.setContentsMargins(0, 0, 0, 0)
        _pb.setSpacing(2)
        self.refine_contour_balloon_sb = QDoubleSpinBox()
        self.refine_contour_balloon_sb.setRange(-1.0, 1.0)
        self.refine_contour_balloon_sb.setSingleStep(0.1)
        self.refine_contour_balloon_sb.setDecimals(2)
        self.refine_contour_balloon_sb.setValue(0.0)
        self.refine_contour_balloon_sb.setToolTip(
            "Pushes the outline out (+) or pulls it in (-) where there is no "
            "wall. It only acts where the wall picture is fainter than the "
            "Edge gate allows - the line below says where.")
        _pb.addLayout(self._refine_row(
            "Push outward", self.refine_contour_balloon_sb))
        self.refine_contour_push_line = QLabel("")
        self.refine_contour_push_line.setWordWrap(True)
        self.refine_contour_push_line.setStyleSheet(_HELPER_LINE_STYLE)
        _pb.addWidget(self.refine_contour_push_line)
        _lo.addWidget(self._refine_contour_push_box)

        # The two weights -- Inside vs outside brightness only.
        self._refine_contour_weights_box = QWidget()
        _gb = QVBoxLayout(self._refine_contour_weights_box)
        _gb.setContentsMargins(0, 0, 0, 0)
        _gb.setSpacing(2)
        self.refine_lambda1_sb = QDoubleSpinBox()
        self.refine_lambda1_sb.setRange(0.1, 10.0)
        self.refine_lambda1_sb.setSingleStep(0.1)
        self.refine_lambda1_sb.setDecimals(1)
        self.refine_lambda1_sb.setValue(1.0)
        # B3: the old tooltips were INVERTED.  A voxel joins the inside
        # when lambda1*(I-c_in)^2 < lambda2*(I-c_out)^2 (refine.
        # _chan_vese_in_band), so a bigger inside weight makes the inside
        # harder to join -- measured on a low-contrast disk: 4598 voxels at
        # 1/1, 180 at 4/1, 10412 at 1/4.
        self.refine_lambda1_sb.setToolTip(
            "Raise it to make the object smaller: a voxel has to look more "
            "like the inside to stay in. Lower it if the object shrinks away.")
        _gb.addLayout(self._refine_row(
            "Weight inside", self.refine_lambda1_sb))
        self.refine_lambda2_sb = QDoubleSpinBox()
        self.refine_lambda2_sb.setRange(0.1, 10.0)
        self.refine_lambda2_sb.setSingleStep(0.1)
        self.refine_lambda2_sb.setDecimals(1)
        self.refine_lambda2_sb.setValue(1.0)
        self.refine_lambda2_sb.setToolTip(
            "Raise it to make the object bigger: a voxel has to look more "
            "like the outside to stay out. Lower it if it bleeds outward.")
        _gb.addLayout(self._refine_row(
            "Weight outside", self.refine_lambda2_sb))
        self._refine_contour_weights_box.setVisible(False)
        _lo.addWidget(self._refine_contour_weights_box)

        self.refine_contour_mode_combo.currentIndexChanged.connect(
            self._on_refine_contour_mode_changed)
        self.refine_contour_axis_combo.currentIndexChanged.connect(
            self._on_refine_contour_axis_changed)
        self.refine_contour_slices_combo.currentIndexChanged.connect(
            self._on_refine_contour_slices_changed)
        self.refine_contour_axis_combo.currentIndexChanged.connect(
            self._refresh_refine_readouts)
        self.refine_contour_balloon_sb.valueChanged.connect(
            self._refresh_refine_push_line)
        return w

    def _build_refine_page_dilate_erode(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.refine_de_radius_sb = QSpinBox()
        self.refine_de_radius_sb.setRange(-10, 10)
        self.refine_de_radius_sb.setValue(2)
        self.refine_de_radius_sb.setToolTip(HELP_LINES["refine_de_radius_sb"])
        lay.addLayout(self._refine_row("Radius", self.refine_de_radius_sb))
        lay.addWidget(self._helper_line("refine_de_radius_sb"))
        self.refine_de_axis_combo = QComboBox()
        self.refine_de_axis_combo.addItems(["3D", "XY", "XZ", "YZ"])
        self.refine_de_axis_combo.setToolTip(
            HELP_LINES["refine_de_axis_combo"])
        lay.addLayout(self._refine_row(
            "Orientation", self.refine_de_axis_combo))
        lay.addWidget(self._helper_line("refine_de_axis_combo"))
        self.refine_de_um_label = QLabel("")
        self.refine_de_um_label.setWordWrap(True)
        self.refine_de_um_label.setStyleSheet(_HELPER_LINE_STYLE)
        lay.addWidget(self.refine_de_um_label)
        self.refine_de_radius_sb.valueChanged.connect(
            self._refresh_refine_readouts)
        self.refine_de_axis_combo.currentIndexChanged.connect(
            self._refresh_refine_readouts)
        return w

    def _build_refine_page_guided_grow(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.refine_gg_max_radius_sb = QSpinBox()
        # 1-15, not 1-20: `_snapshot_label_roi` pads the crop
        # by 15, so at 20 the audit grew 6,199 voxels into the ROI WALL.
        self.refine_gg_max_radius_sb.setRange(1, 15)
        self.refine_gg_max_radius_sb.setValue(5)
        self.refine_gg_max_radius_sb.setToolTip(
            HELP_LINES["refine_gg_max_radius_sb"])
        lay.addLayout(self._refine_row(
            "Max radius", self.refine_gg_max_radius_sb))
        lay.addWidget(self._helper_line("refine_gg_max_radius_sb"))
        self.refine_gg_prob_threshold_sb = QDoubleSpinBox()
        self.refine_gg_prob_threshold_sb.setRange(0.0, 1.0)
        self.refine_gg_prob_threshold_sb.setSingleStep(0.05)
        self.refine_gg_prob_threshold_sb.setDecimals(2)
        self.refine_gg_prob_threshold_sb.setValue(0.50)
        # `guided_grow` rescales the map to 0-1 over THIS object's region
        # before comparing, so the number is relative to what is around it.
        self.refine_gg_prob_threshold_sb.setToolTip(
            HELP_LINES["refine_gg_prob_threshold_sb"])
        lay.addLayout(self._refine_row(
            "Confidence needed", self.refine_gg_prob_threshold_sb))
        lay.addWidget(self._helper_line("refine_gg_prob_threshold_sb"))
        # `refine_gg_source_combo` (Probability / Image / Both) is RETIRED
        # here, not merely hidden: the image mode admitted 82 % of the
        # region and leaked through an open top (measured), and
        # the orchestrator no longer passes an `image_roi` at all.
        self.refine_gg_axis_combo = QComboBox()
        self.refine_gg_axis_combo.addItems(["3D", "XY", "XZ", "YZ"])
        self.refine_gg_axis_combo.setToolTip(
            HELP_LINES["refine_gg_axis_combo"])
        lay.addLayout(self._refine_row(
            "Orientation", self.refine_gg_axis_combo))
        lay.addWidget(self._helper_line("refine_gg_axis_combo"))
        return w

    def _build_refine_page_fill_holes(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.refine_fh_axis_combo = QComboBox()
        self.refine_fh_axis_combo.addItems(["3D", "XY", "XZ", "YZ"])
        # B3: the old tooltip had this BACKWARDS.  A cavity sealed in 3D
        # is sealed in every slice through it, so a plane mode fills
        # everything 3D does and more (`fill_holes_label`).
        self.refine_fh_axis_combo.setToolTip(
            HELP_LINES["refine_fh_axis_combo"])
        lay.addLayout(self._refine_row(
            "Orientation", self.refine_fh_axis_combo))
        lay.addWidget(self._helper_line("refine_fh_axis_combo"))
        return w

    def _build_refine_page_flood_fill(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.refine_ff_axis_combo = QComboBox()
        self.refine_ff_axis_combo.addItems(["3D", "XY", "XZ", "YZ"])
        self.refine_ff_axis_combo.setToolTip(
            HELP_LINES["refine_ff_axis_combo"])
        lay.addLayout(self._refine_row(
            "Orientation", self.refine_ff_axis_combo))
        lay.addWidget(self._helper_line("refine_ff_axis_combo"))
        self.refine_ff_max_size_sb = QSpinBox()
        self.refine_ff_max_size_sb.setRange(100, 500000)
        self.refine_ff_max_size_sb.setSingleStep(1000)
        self.refine_ff_max_size_sb.setValue(50000)
        self.refine_ff_max_size_sb.setToolTip(
            HELP_LINES["refine_ff_max_size_sb"])
        lay.addLayout(self._refine_row(
            "Max size", self.refine_ff_max_size_sb))
        lay.addWidget(self._helper_line("refine_ff_max_size_sb"))
        desc = QLabel(
            "Put the crosshair in the gap you want filled, on or next to "
            "this object, then press Preview."
        )
        desc.setStyleSheet(_HELPER_LINE_STYLE)
        desc.setWordWrap(True)
        lay.addWidget(desc)
        return w

    def _build_refine_page_settle(self) -> QWidget:
        """Settle onto the wall signal."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # "Wall type" is NOT on this page any more (B2): Settle and the
        # contour's *Snap to edges* read the SAME wall kind, so its one combo
        # lives in the shared block under the channel card
        # (`_build_refine_wall_row`), next to the channel it is read from.

        solver_row = QHBoxLayout()
        solver_row.setSpacing(6)
        solver_row.addWidget(panel_form_label("Solve with", parent=self))
        self.refine_solver_combo = QComboBox()
        self.refine_solver_combo.addItems(list(REFINE_SOLVER_LABELS))
        self._apply_item_tooltips("refine_solver_combo")
        self.refine_solver_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_solver_combo.setToolTip(HELP_LINES["refine_solver_combo"])
        solver_row.addWidget(self.refine_solver_combo, 1)
        lay.addLayout(solver_row)
        lay.addWidget(self._helper_line("refine_solver_combo"))
        # The slot `set_refine_solver_available` writes the "not installed"
        # reason into.  Empty and hidden unless it has something to say.
        # Its OWN attribute (B3): parked in `_item_help_labels` it would be
        # overwritten the day anything refreshed the solver's item help.
        _solver_note = QLabel("")
        _solver_note.setWordWrap(True)
        _solver_note.setStyleSheet(PANEL_WARN_STYLE)
        _solver_note.setVisible(False)
        self._refine_solver_note = _solver_note
        lay.addWidget(_solver_note)

        band_row = QHBoxLayout()
        band_row.setSpacing(6)
        band_row.addWidget(panel_form_label("Edge may move", parent=self))
        self.refine_settle_band_sb = QDoubleSpinBox()
        self.refine_settle_band_sb.setRange(0.5, 20.0)
        self.refine_settle_band_sb.setSingleStep(0.5)
        self.refine_settle_band_sb.setDecimals(1)
        self.refine_settle_band_sb.setValue(2.0)
        self.refine_settle_band_sb.setSuffix(" um")
        self.refine_settle_band_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_settle_band_sb.setMinimumWidth(
            self._readable_spin_width(self.refine_settle_band_sb))
        self.refine_settle_band_sb.setToolTip(
            HELP_LINES["refine_settle_band_sb"])
        band_row.addWidget(self.refine_settle_band_sb, 1)
        lay.addLayout(band_row)
        lay.addWidget(self._helper_line("refine_settle_band_sb"))

        iter_row = QHBoxLayout()
        iter_row.setSpacing(6)
        iter_row.addWidget(panel_form_label("Steps", parent=self))
        self.refine_settle_iter_sb = QSpinBox()
        self.refine_settle_iter_sb.setRange(1, 50)
        self.refine_settle_iter_sb.setValue(10)
        self.refine_settle_iter_sb.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_settle_iter_sb.setMinimumWidth(
            self._readable_spin_width(self.refine_settle_iter_sb))
        self.refine_settle_iter_sb.setToolTip(
            HELP_LINES["refine_settle_iter_sb"])
        iter_row.addWidget(self.refine_settle_iter_sb, 1)
        lay.addLayout(iter_row)
        lay.addWidget(self._helper_line("refine_settle_iter_sb"))

        return w

    def _build_refine_wall_row(self) -> QWidget:
        """The ONE wall-kind picker, shared by Settle and *Snap to edges*.

        It sits in the shared block, directly under the channel card it
        reads from -- which is what "the wall rows of the channel card" are
       .  Shown only while the chosen method builds a wall
        picture, so it disappears for *Inside vs outside brightness* (trim
        2).  Its grey line shows on Settle only: on the contour page it
        would repeat what the mode line already says (trim 4).
        """
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        wall_row = QHBoxLayout()
        wall_row.setSpacing(6)
        wall_row.addWidget(panel_form_label("Wall type", parent=self))
        self.refine_wall_kind_combo = QComboBox()
        self.refine_wall_kind_combo.addItems(list(REFINE_WALLS_FROM_LABELS))
        self._apply_item_tooltips("refine_wall_kind_combo")
        self.refine_wall_kind_combo.setCurrentIndex(
            REFINE_WALL_KINDS.index("membrane"))
        self.refine_wall_kind_combo.setStyleSheet(PANEL_INPUT_STYLE)
        self.refine_wall_kind_combo.setToolTip(
            HELP_LINES["refine_wall_kind_combo"])
        wall_row.addWidget(self.refine_wall_kind_combo, 1)
        lay.addLayout(wall_row)
        self._refine_wall_help = self._helper_line("refine_wall_kind_combo")
        lay.addWidget(self._refine_wall_help)
        # Settle + a shape wall (B3 audit): amber, on the page, only while
        # that combination is chosen.
        self.refine_shape_wall_note = QLabel(REFINE_SHAPE_WALL_NOTE)
        self.refine_shape_wall_note.setWordWrap(True)
        self.refine_shape_wall_note.setStyleSheet(PANEL_WARN_STYLE)
        self.refine_shape_wall_note.setVisible(False)
        lay.addWidget(self.refine_shape_wall_note)
        self.refine_wall_kind_combo.currentIndexChanged.connect(
            self._on_refine_wall_kind_changed)
        return box

    # --- Refine method descriptions ----------------------------------------
    #
    # Keyed 0..6 to match `REFINE_METHOD_KEYS`.  `_on_refine_method_changed`
    # reads them by the combo's index, so they move with it.

    _REFINE_METHOD_DESCRIPTIONS = {
        0: (  # Smooth (signed distance)
            "Rounds the object off: it is turned into a distance field "
            "(how far each voxel is from the outline), blurred, and re-cut "
            "at the halfway line.\n"
            "This is what takes the slice-by-slice staircase out of a "
            "hand-drawn object. The two numbers are in VOXELS, not microns, "
            "so on coarse depth a small number is a long way."
        ),
        # Trim 1: the line under Mode describes the CHOSEN mode,
        # so this one does not describe both -- that sentence is the mode
        # combo's tooltip.  The tallest page on the tab pays for every line.
        1: (  # Active contour
            "Slides the outline onto the image, in 3D or slice by slice."
        ),
        2: (  # Dilate/Erode
            "Grows (+) or shrinks (-) the object by a fixed number of "
            "voxels.\n"
            "Growing never takes a voxel from a neighbouring object. Keep it "
            "to one plane for an orientation-specific fix."
        ),
        3: (  # Guided Grow
            "Grows the object outward into empty voxels the reconstruction "
            "thought belonged to something.\n"
            "The best tool for closing an open surface the reconstruction "
            "left a gap in."
        ),
        4: (  # Fill Holes
            "Closes cavities inside the object.\n"
            "3D closes only a void sealed on every side. A plane mode also "
            "closes a hole sealed within one slice but open above or below, "
            "so it fills more, not less."
        ),
        5: (  # Flood Fill
            "Takes the connected empty space at the crosshair and gives it "
            "to this object.\n"
            "Put the crosshair in the void first, using the three views."
        ),
        6: (  # Settle onto the wall signal
            "Moves the outline onto the nearest wall, and no further than "
            "you allow.\n"
            "Neighbouring objects take part, so the gap between two objects "
            "is divided on the wall rather than swallowed - and no other "
            "object ever loses a voxel. An edge that already touches a "
            "neighbour does not move."
        ),
    }

    # --- Refine page helpers ------------------------------------------------

    def refine_method_key(self) -> str:
        """The ENGINE KEY of the method the combo is showing."""
        try:
            idx = int(self.refine_method_combo.currentIndex())
        except Exception:
            return "sdf"
        for i, _text, key in REFINE_METHOD_KEYS:
            if i == idx:
                return key
        return "sdf"

    def refine_is_wall_method(self) -> bool:
        """Does the chosen method build a wall ELEVATION?

        Settle always does.  Active contour does in *Snap to edges* and
        does NOT in *Inside vs outside brightness*, which reads the
        prepared channel's intensity directly -- so that mode keeps the
        "Local contrast" tick, and the wall-building ones drop it (the
        ``drop_clahe`` consumer rule).
        """
        key = self.refine_method_key()
        if key == "settle":
            return True
        if key == "contour":
            try:
                return int(self.refine_contour_mode_combo.currentIndex()) == 0
            except Exception:
                return True
        return False

    def refine_wall_kind(self) -> str:
        """The engine's wall-kind token, read BY INDEX from the combo."""
        combo = getattr(self, "refine_wall_kind_combo", None)
        try:
            idx = int(combo.currentIndex())
        except Exception:
            return "membrane"
        if 0 <= idx < len(REFINE_WALL_KINDS):
            return REFINE_WALL_KINDS[idx]
        return "membrane"

    def refine_needs_channel(self) -> bool:
        """Does the CURRENT method+wall combination read an image channel?

        Every method except Active contour and Settle is no; Settle on a
        wall kind that is not image-derived (the two shape kinds and the
        probability map) is also no, and asking for a channel that will
        not be read is the same mistake in the other direction.
        """
        key = self.refine_method_key()
        if key not in REFINE_CHANNEL_METHODS:
            return False
        if key == "contour":
            # *Inside vs outside brightness* reads the prepared channel
            # directly, so it always needs one.  *Snap to edges* builds a
            # wall elevation, so it needs a channel on exactly the same
            # terms Settle does -- and the two shape kinds and the
            # probability map need none (audit C-M3: this branch used to
            # return True unconditionally, so the contour asked for a
            # channel it would not read and refused without one).
            try:
                if int(self.refine_contour_mode_combo.currentIndex()) == 1:
                    return True
            except Exception:
                return True
        try:
            idx = int(self.refine_wall_kind_combo.currentIndex())
        except Exception:
            return True
        return idx in REFINE_IMAGE_WALL_INDICES

    def _on_refine_about_toggled(self, checked: bool) -> None:
        self.refine_about_btn.setText(
            _CHEVRON_OPEN if checked else _CHEVRON_CLOSED)
        self._refine_about_body.setVisible(bool(checked))

    def _on_refine_advanced_toggled(self, checked: bool) -> None:
        self._refine_advanced_container.setVisible(bool(checked))
        self._refresh_chevron_text(self.refine_advanced_btn)

    def _on_refine_prep_toggled(self, checked: bool) -> None:
        self._refine_prep_container.setVisible(bool(checked))
        self._refresh_chevron_text(self.refine_prep_btn)

    def _on_refine_contour_mode_changed(self, index: int) -> None:
        self._refresh_item_help("refine_contour_mode_combo")
        self._apply_refine_method_visibility()

    def _on_refine_contour_axis_changed(self, _index: int) -> None:
        self._apply_refine_method_visibility()

    def _on_refine_contour_slices_changed(self, index: int) -> None:
        self._refine_contour_share_row.setVisible(int(index) == 0)
        self._refine_contour_nslices_row.setVisible(int(index) == 1)
        self._refresh_item_help("refine_contour_slices_combo")

    def refine_contour_is_2d(self) -> bool:
        combo = getattr(self, "refine_contour_axis_combo", None)
        try:
            return int(combo.currentIndex()) > 0
        except Exception:
            return False

    def refine_contour_push_text(self):
        """``(text, no_effect)`` -- the LIVE line under *Push outward*.

        Recomputed from the push, the gate and the sharpness: a number that does nothing must say so while the user is
        looking at it, not after a five-second Preview.
        """
        from packages.epicycle_annotator.refine import push_acts_below

        b = float(self.refine_contour_balloon_sb.value())
        g = float(self.refine_contour_gate_sb.value())
        k = float(self.refine_contour_sharpness_sb.value())
        if b == 0.0:
            return ("No push: the outline follows the walls alone.", False)
        below = push_acts_below(b, g, k)
        if below is None:
            return ("No effect at this setting - the push is not bigger "
                    "than the Edge gate ({:.2f}). Push harder, or lower the "
                    "gate on the Advanced card.".format(g), True)
        if below == float("inf") or below >= 1.0:
            # the wall picture never exceeds 1, so a threshold at or above
            # 1 lets the push act on every voxel (B2 audit UX-L)
            return ("Acts everywhere at this setting.", False)
        return ("Acts where the wall picture is fainter than {:.2f}."
                .format(below), False)

    def _refresh_refine_push_line(self, *_a) -> None:
        line = getattr(self, "refine_contour_push_line", None)
        if line is None:
            return
        text, no_effect = self.refine_contour_push_text()
        line.setText(text)
        line.setStyleSheet(PANEL_WARN_STYLE if no_effect
                           else _HELPER_LINE_STYLE)

    def _on_refine_wall_kind_changed(self, _index: int) -> None:
        self._apply_refine_method_visibility()

    def _prep_ticks_refine(self) -> dict:
        return self._prep_ticks("refine_prep")

    def refine_prep_chain(self) -> str:
        """The chain the CURRENT method gets.

        One ticklist, one rule: a method that builds a wall elevation takes
        it with ``drop_clahe=True``, because the wall stage applies its own
        CLAHE through the ``membrane clip`` knob and a second one would
        double-apply it.  That is also why the "Local contrast" tick is
        HIDDEN on those methods rather than silently ignored.
        """
        return chain_from_ticks(
            self._prep_ticks("refine_prep"),
            drop_clahe=self.refine_is_wall_method(),
        )

    def _refresh_refine_prep_chip(self, *_a) -> None:
        btn = getattr(self, "refine_prep_btn", None)
        if btn is None:
            return
        self._set_chevron_title(
            btn, "Prepare the channel first: {}".format(
                self._chain_summary(self.refine_prep_chain()))
        )

    def _apply_refine_method_visibility(self) -> None:
        """Show exactly the rows the chosen method reads.

        The channel row and the prepare chevron are HIDDEN -- not disabled
        -- for a method that does not read a channel, and the "pick a
        channel" rule does not apply while they are.
        """
        key = self.refine_method_key()
        needs = self.refine_needs_channel()
        is_prob = (self.refine_is_wall_method()
                   and self.refine_wall_kind() == "probability")
        card = getattr(self, "_refine_channel_container", None)
        if card is not None:
            card.setVisible(bool(needs))
        line = getattr(self, "_refine_no_channel_line", None)
        if line is not None:
            line.setVisible(bool(is_prob))
        # "Local contrast" is not offered where the tool is looking for
        # walls, because the wall stage applies its own.
        row = (getattr(self, "_refine_prep_rows", None) or {}).get("clahe")
        if row is not None:
            wallish = self.refine_is_wall_method()
            for w in row:
                w.setVisible(not wallish)
            # ...and the card says so where the tick is missing.
            no_clahe = getattr(self, "_refine_prep_no_clahe_line", None)
            if no_clahe is not None:
                no_clahe.setVisible(bool(wallish))
        adv = getattr(self, "_refine_settle_advanced", None)
        if adv is not None:
            adv.setVisible(key == "settle")
        # The shared wall row: shown exactly while the method builds a wall
        # picture (Settle, Snap to edges) -- trim 2 takes it away for the
        # brightness mode.  Its grey line on Settle only (trim 4).
        wall_box = getattr(self, "_refine_wall_container", None)
        if wall_box is not None:
            wall_box.setVisible(bool(self.refine_is_wall_method()))
        wall_help = getattr(self, "_refine_wall_help", None)
        if wall_help is not None:
            wall_help.setVisible(key == "settle")
        shape_note = getattr(self, "refine_shape_wall_note", None)
        if shape_note is not None:
            shape_note.setVisible(
                key == "settle"
                and self.refine_wall_kind() in REFINE_SHAPE_WALL_KINDS)
        # The contour's own rows.
        if getattr(self, "refine_contour_mode_combo", None) is not None:
            snap = int(self.refine_contour_mode_combo.currentIndex()) == 0
            two_d = self.refine_contour_is_2d()
            self._refine_contour_push_box.setVisible(snap)
            self._refine_contour_weights_box.setVisible(not snap)
            self._refine_contour_sampling.setVisible(two_d)
            self._refine_contour_advanced.setVisible(key == "contour")
            self._refine_contour_lower.setVisible(key == "contour")
            self._refine_contour_snap_adv.setVisible(snap)
            self._refine_contour_rebuild_adv.setVisible(two_d)
            self._on_refine_contour_slices_changed(
                self.refine_contour_slices_combo.currentIndex())
            self._refresh_refine_push_line()
        self._refresh_refine_prep_chip()
        self._refresh_refine_readouts()
        # The stack answers for the page it SHOWS, so every change of what
        # the page shows must tell the card: the card's layout item caches
        # the stack's height-for-width and would otherwise keep the LAST
        # page's (measured: the contour laid out at the Smooth page's
        # 201 px with 47 px of rows).
        stack = getattr(self, "_refine_params_stack", None)
        if stack is not None:
            stack.updateGeometry()
            card = getattr(self, "_refine_method_card", None)
            if card is not None and card.layout() is not None:
                card.layout().invalidate()

    def _refresh_refine_readouts(self, *_a) -> None:
        """The micron read-outs beside the voxel knobs.

        Suppressed entirely when the controller could not resolve a voxel
        size: printing a wrong number is worse than printing none, and S5
        carries the amber note that says why.
        """
        spacing = getattr(self, "_refine_spacing", None)
        sdf = getattr(self, "refine_sdf_um_label", None)
        de = getattr(self, "refine_de_um_label", None)
        join = getattr(self, "refine_contour_rebuild_readout", None)
        if join is not None:
            ax = 0
            try:
                ax = max(0, int(self.refine_contour_axis_combo.currentIndex())
                         - 1)
            except Exception:
                ax = 0
            if spacing:
                deep = float(spacing[ax]) or 1.0
                join.setText("About {:.1f} voxels deep on this patch.".format(
                    float(self.refine_contour_rebuild_um_sb.value()) / deep))
                join.setVisible(True)
            else:
                join.setText("")
                join.setVisible(False)
        if not spacing:
            for lb in (sdf, de):
                if lb is not None:
                    lb.setText("")
                    lb.setVisible(False)
            return
        z, y, x = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        if sdf is not None:
            #  the wording the owner reads on a thin object:
            # the product is printed with ONE decimal ("about 9.0 um"), so
            # the number that explains a deletion reads as a measurement.
            sdf.setText(
                "These are counted in voxels, not microns. On this patch one "
                "voxel through depth is {:.3g} um, so {:.1f} blurs over about "
                "{:.1f} um; across a slice {:.1f} is about {:.1f} um.".format(
                    z, self.smooth_standalone_sigma_z_sb.value(),
                    z * self.smooth_standalone_sigma_z_sb.value(),
                    self.smooth_standalone_sigma_xy_sb.value(),
                    x * self.smooth_standalone_sigma_xy_sb.value())
            )
            sdf.setVisible(True)
        if de is not None:
            r = abs(int(self.refine_de_radius_sb.value()))
            try:
                de_axis = str(self.refine_de_axis_combo.currentText())
            except Exception:
                de_axis = "3D"
            if de_axis == "XY":
                # XY never moves through depth (B3 audit LOW), so a
                # through-depth number would describe nothing it does.
                de.setText("In microns that is about {:.3g} across a "
                           "slice.".format(r * x))
            else:
                de.setText(
                    "In microns that is about {:.3g} through depth and "
                    "{:.3g} across a slice.".format(r * z, r * x)
                )
            de.setVisible(True)

    def set_refine_prob_overlay_message(self, text=None) -> None:
        """Say ON SCREEN why the probability overlay is not showing.

        A non-empty ``text`` also takes the tick back off, silently (under
        ``blockSignals``, so the annotator is not asked to remove a layer it
        never added).  ``None`` or ``""`` clears the line.
        """
        text = str(text or "")
        lb = getattr(self, "refine_prob_overlay_msg", None)
        if lb is not None:
            lb.setText(text)
            lb.setVisible(bool(text))
        cb = getattr(self, "refine_prob_overlay_cb", None)
        if text and cb is not None and cb.isChecked():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def set_refine_spacing(self, spacing) -> None:
        """Tell the page the patch's voxel size, or ``None`` for unknown."""
        self._refine_spacing = tuple(spacing) if spacing else None
        self._refresh_refine_readouts()

    def set_refine_channels(self, names, current: str = "") -> None:
        """Fill the Refine channel combo -- the SIXTH channel combo.

        The shared five-point policy, verbatim: the last name
        actually CHOSEN is remembered in a Qt property rather than read
        back off ``currentText()``, and a restore miss leaves index -1 so
        the combo stays blank and the explicit refusal stays reachable.
        """
        combo = getattr(self, "refine_channel_combo", None)
        if combo is None:
            return
        prev = current or self._remember_channel_choice(combo)
        combo.blockSignals(True)
        combo.clear()
        for name in names or ():
            combo.addItem(str(name))
        self._restore_channel_choice(combo, prev)
        if not prev:
            # First fill with nothing remembered: stay blank (as Add does)
            # so the "pick a channel" refusal fires instead of a silent
            # guess of the first image layer.
            combo.setCurrentIndex(-1)
        combo.blockSignals(False)

    def set_refine_guided_grow_available(self, available: bool,
                                         reason: str = "") -> None:
        """Grey Guided Grow when the patch has no probability map.

        The ITEM, not the combo, exactly as the diffusion solver's item is
        greyed by `set_refine_solver_available`: the other six methods are
        still perfectly usable.  The item's own text says why, so the reason
        is on screen the moment the list opens.
        Today the same case used to fall through to the retired intensity
        band; B1 turned it into a refusal AFTER the click, which taught
        nothing until the user had already asked.
        """
        combo = getattr(self, "refine_method_combo", None)
        if combo is None:
            return
        idx = REFINE_METHOD_INDEX["guided_grow"]
        text = next(t for i, t, _k in REFINE_METHOD_KEYS if i == idx)
        try:
            item = combo.model().item(idx)
        except Exception:
            item = None
        reason = str(reason or REFINE_GUIDED_GROW_NO_PROB)
        if item is not None:
            item.setEnabled(bool(available))
            item.setToolTip(
                self._item_tooltip("refine_method_combo", idx)
                if available else reason)
            item.setText(text if available else REFINE_GUIDED_GROW_ITEM_NO_PROB)
        if not available and int(combo.currentIndex()) == idx:
            combo.setCurrentIndex(0)
        # The line under the combo that used to repeat the reason is
        # DROPPED (B2 audit UX-M4, pending the owner): on a patch with no
        # map it cost every page 26 px, and the greyed item's own text is
        # on screen the moment the list opens.

    def refine_guided_grow_available(self) -> bool:
        combo = getattr(self, "refine_method_combo", None)
        try:
            item = combo.model().item(REFINE_METHOD_INDEX["guided_grow"])
            return bool(item.isEnabled())
        except Exception:
            return True

    def set_refine_solver_available(self, available: bool,
                                    reason: str = "") -> None:
        """Grey the diffusion solver when u-Segment3D is absent.

        The ITEM, not the combo: the other two solvers are still perfectly
        usable, and a whole picker greyed for one missing dependency is a
        picker the user cannot read the reason off.  The reason is ALSO put
        under the combo, because Qt shows no tooltip on a disabled item.

        Driven from the annotator rather than from `__init__`, so the
        optional import is paid once at dock launch instead of on every
        panel build.
        """
        combo = getattr(self, "refine_solver_combo", None)
        if combo is None:
            return
        idx = REFINE_SOLVERS.index("label_diffusion")
        try:
            item = combo.model().item(idx)
        except Exception:
            item = None
        if item is not None:
            item.setEnabled(bool(available))
            item.setToolTip(
                self._item_tooltip("refine_solver_combo", idx)
                if available else str(reason))
        if not available and int(combo.currentIndex()) == idx:
            combo.setCurrentIndex(0)
        line = getattr(self, "_refine_solver_note", None)
        if line is None:
            return
        line.setText("" if available else str(reason))
        line.setVisible(not available)

    def refine_channel_name(self) -> str:
        """The chosen channel, or ``""`` -- never a fallback."""
        combo = getattr(self, "refine_channel_combo", None)
        try:
            if combo is None or combo.currentIndex() < 0:
                return ""
            return str(combo.currentText() or "")
        except Exception:
            return ""

    def get_refine_params(self) -> dict:
        """Everything the orchestrator needs to run the chosen method.

        Every combo is read BY INDEX and mapped through a module constant,
        so relabelling a control is safe by construction and REORDERING one
        is a breaking change that has to move its constant with it.
        """
        def _val(attr, default):
            w = getattr(self, attr, None)
            try:
                return w.value()
            except Exception:
                return default

        def _chk(attr, default=False):
            w = getattr(self, attr, None)
            try:
                return bool(w.isChecked())
            except Exception:
                return default

        def _idx(attr, default=0):
            w = getattr(self, attr, None)
            try:
                return int(w.currentIndex())
            except Exception:
                return default

        solver_idx = _idx("refine_solver_combo", 0)
        return {
            "method": self.refine_method_key(),
            "channel": self.refine_channel_name(),
            "prep_chain": self.refine_prep_chain(),
            "needs_channel": self.refine_needs_channel(),
            "wall_kind": self.refine_wall_kind(),
            "solver": (REFINE_SOLVERS[solver_idx]
                       if 0 <= solver_idx < len(REFINE_SOLVERS)
                       else REFINE_SOLVERS[0]),
            "band_um": float(_val("refine_settle_band_sb", 2.0)),
            "n_iter": int(_val("refine_settle_iter_sb", 10)),
            "clamp": float(_val("refine_settle_clamp_sb", 0.5)),
            "compactness": float(_val("refine_settle_compactness_sb", 0.0)),
            "contour_mode": _idx("refine_contour_mode_combo", 0),
            # B2 -- the merged Active contour.  ADDITIVE keys only: every
            # key above keeps its meaning.  Combos by INDEX through the
            # module constants.
            "contour_axis": REFINE_CONTOUR_AXES[
                min(max(_idx("refine_contour_axis_combo", 0), 0),
                    len(REFINE_CONTOUR_AXES) - 1)],
            "contour_sampling": min(max(
                _idx("refine_contour_slices_combo", 0), 0), 2),
            "contour_share_pct": float(_val("refine_contour_share_sb", 25)),
            "contour_n_slices": int(_val("refine_contour_nslices_sb", 12)),
            "contour_limit_um": float(_val("refine_contour_limit_sb", 3.0)),
            "contour_iterations": int(
                _val("refine_contour_iterations_sb", 100)),
            "contour_balloon": float(_val("refine_contour_balloon_sb", 0.0)),
            "contour_gate": float(_val("refine_contour_gate_sb", 0.30)),
            "contour_blur_um": float(_val("refine_contour_blur_sb", 2.0)),
            "contour_sharpness": float(
                _val("refine_contour_sharpness_sb", 100)),
            "contour_rebuild_um": float(
                _val("refine_contour_rebuild_um_sb", 1.5)),
            "contour_lambda1": float(_val("refine_lambda1_sb", 1.0)),
            "contour_lambda2": float(_val("refine_lambda2_sb", 1.0)),
            "sigma_z": float(_val("smooth_standalone_sigma_z_sb", 3.0)),
            "sigma_xy": float(_val("smooth_standalone_sigma_xy_sb", 1.5)),
            "preserve_topology": _chk("smooth_preserve_topology_cb", True),
        }

    # --- Refine recipes ---------
    #
    # One step's settings as the recipe stores them: every combo as its
    # ENGINE TOKEN through its module constant (never an index), every
    # number as the widget shows it.  `get_refine_params()` is untouched --
    # these two read and write the widgets beside it.

    #: The widgets behind each method's numbers, ``param -> attribute``.
    _RECIPE_SPINS = {
        "sdf": {"sigma_z": "smooth_standalone_sigma_z_sb",
                "sigma_xy": "smooth_standalone_sigma_xy_sb"},
        "contour": {"share_pct": "refine_contour_share_sb",
                    "n_slices": "refine_contour_nslices_sb",
                    "limit_um": "refine_contour_limit_sb",
                    "iterations": "refine_contour_iterations_sb",
                    "balloon": "refine_contour_balloon_sb",
                    "gate": "refine_contour_gate_sb",
                    "blur_um": "refine_contour_blur_sb",
                    "sharpness": "refine_contour_sharpness_sb",
                    "rebuild_um": "refine_contour_rebuild_um_sb",
                    "lambda1": "refine_lambda1_sb",
                    "lambda2": "refine_lambda2_sb"},
        "dilate_erode": {"radius": "refine_de_radius_sb"},
        "guided_grow": {"max_radius": "refine_gg_max_radius_sb",
                        "prob_threshold": "refine_gg_prob_threshold_sb"},
        "fill_holes": {},
        "settle": {"band_um": "refine_settle_band_sb",
                   "n_iter": "refine_settle_iter_sb",
                   "clamp": "refine_settle_clamp_sb",
                   "compactness": "refine_settle_compactness_sb"},
        "flood_fill": {"max_size": "refine_ff_max_size_sb"},
    }

    #: The per-method axis combos, set by TEXT (``findText``; a miss is an
    #: error, never index 0).
    _RECIPE_AXIS_COMBO = {
        "dilate_erode": "refine_de_axis_combo",
        "guided_grow": "refine_gg_axis_combo",
        "fill_holes": "refine_fh_axis_combo",
        "flood_fill": "refine_ff_axis_combo",
    }

    def populate_refine_recipes_menu(self, entries) -> None:
        """Render ``refine_recipe.menu_entries`` -- nothing is decided here.

        ``entries`` is ``[(kind, text, enabled, name, tooltip)]``.  Every
        greyed item's reason IS its text; a tooltip carries an
        elided summary's full chain.  Actions are rebuilt each time, so a
        triggered connection can never accumulate.
        """
        menu = getattr(self, "refine_recipes_menu", None)
        if menu is None:
            return
        menu.clear()
        cbs = getattr(self, "_refine_recipe_cbs", {}) or {}
        subs = {}

        def _sub(kind):
            if kind not in subs:
                title = ("Load into the panel" if kind == "load"
                         else "Delete")
                subs[kind] = menu.addMenu(title)
                subs[kind].setToolTipsVisible(True)
            return subs[kind]

        for kind, text, enabled, name, tip in entries or ():
            if kind == "separator":
                menu.addSeparator()
                continue
            label = str(text).replace("&", "&&")
            if kind == "run":
                label = "    " + label
            target = _sub(kind) if kind in ("load", "delete") else menu
            act = target.addAction(label)
            act.setEnabled(bool(enabled))
            if tip:
                act.setToolTip(str(tip))
            cb = cbs.get(kind)
            if cb is not None and kind in ("run", "load", "save", "delete"):
                if kind == "save":
                    act.triggered.connect(lambda _c=False, f=cb: f())
                else:
                    act.triggered.connect(
                        lambda _c=False, f=cb, n=name: f(n))
        # Both sub-menus are always offered, greyed when they would be empty.
        if entries and not (len(entries) == 1 and entries[0][0] == "info"):
            for kind in ("load", "delete"):
                if kind not in subs:
                    _sub(kind).setEnabled(False)

    def _apply_refine_stage_elision(self) -> None:
        """Right-elide the stage line to the width it has."""
        lb = getattr(self, "refine_stage_label", None)
        if lb is None:
            return
        full = str(getattr(self, "_refine_stage_full_text", "") or "")
        width = lb.contentsRect().width()
        if not lb.isVisible() or width <= 0:
            # Not laid out yet: the full text, re-elided on the next Show /
            # Resize (`_ElideOnResize`).
            lb.setText(full)
            return
        lb.setText(lb.fontMetrics().elidedText(full, Qt.ElideRight, width))

    def get_refine_prep_ticks(self) -> dict:
        """The five Refine prepare ticks, as a plain dict."""
        return self._prep_ticks("refine_prep")

    def refine_solver_available(self) -> bool:
        """Is the diffusion solver's item enabled (u-Segment3D present)?"""
        combo = getattr(self, "refine_solver_combo", None)
        try:
            item = combo.model().item(REFINE_SOLVERS.index("label_diffusion"))
            return bool(item.isEnabled())
        except Exception:
            return True

    def get_refine_step_params(self, method: str) -> dict:
        """The recipe params of ``method`` as the page shows them now.

        ``sdf`` / ``contour`` / ``settle`` go through
        ``refine_recipe.record_from_params`` over ``get_refine_params()``,
        so their ``prep_chain`` is the one the CURRENT method would run --
        read it while that method is on the combo.
        """
        from packages.epicycle_annotator import refine_recipe as rr

        if method in ("sdf", "contour", "settle"):
            return dict(rr.record_from_params(
                method, self.get_refine_params(),
                prep_ticks=self.get_refine_prep_ticks()).params)
        if method not in self._RECIPE_SPINS:
            raise ValueError("unknown Refine method {!r}".format(method))
        out = {}
        for key, attr in self._RECIPE_SPINS[method].items():
            w = getattr(self, attr)
            v = w.value()
            out[key] = int(v) if isinstance(w, QSpinBox) else float(v)
        combo = self._RECIPE_AXIS_COMBO.get(method)
        if combo is not None:
            out["axis"] = str(getattr(self, combo).currentText())
        return out

    def set_refine_step_params(self, method: str, params: dict):
        """Put one recipe step on the page -- ``(ok, reason)``, never raises.

        The inverse of :meth:`get_refine_step_params`.  The
        method combo is set WITH signals on, so the page, the description,
        the channel card and the annotator's own handler all follow.  Every
        other combo by index through its module constant; the axis combos
        by text.  Every number is checked against its widget's range and
        decimals BEFORE anything is set -- Qt clamps and rounds silently,
        and a clamped value is a different recipe.  The channel combo is
        not touched (by design).  A greyed method or solver item is
        refused: a recipe never runs a step the user could not click.
        """
        from packages.epicycle_annotator import refine_recipe as rr
        from packages.epicycle_annotator.refine import (
            CONTOUR_MODES,
            SLICE_SAMPLINGS,
        )

        try:
            step = rr.RecipeStep(str(method), dict(params or {}))
            step.validate()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        p = step.params
        try:
            mcombo = self.refine_method_combo
            midx = REFINE_METHOD_INDEX[method]
            item = mcombo.model().item(midx)
            if item is not None and not item.isEnabled():
                return False, "{} is not available on this patch".format(
                    rr.STEP_WORD.get(method, method))
            if method == "settle" and p["solver"] == "label_diffusion" and (
                    not self.refine_solver_available()):
                return False, "its solver is not available here"
            # Every number first, nothing set yet.
            plan = []
            for key, attr in self._RECIPE_SPINS.get(method, {}).items():
                w = getattr(self, attr)
                v = p[key]
                if not (w.minimum() - 1e-9 <= float(v)
                        <= w.maximum() + 1e-9):
                    return False, "a value is out of range"
                if isinstance(w, QDoubleSpinBox):
                    if abs(round(float(v), w.decimals()) - float(v)) > 1e-9:
                        return False, "a value has more decimals than the box"
                elif float(v) != int(v):
                    return False, "a value is not a whole number"
                plan.append((w, v))
            if "prep_ticks" in p:
                # Audit L2: all-or-nothing -- the chain these ticks give on
                # THIS method is checked before a single widget moves.
                wallish = method == "settle" or (
                    method == "contour" and p["mode"] == "snap")
                if chain_from_ticks(p["prep_ticks"],
                                    drop_clahe=wallish) != p["prep_chain"]:
                    return False, rr.STAGE_PREP_DIFFERS
            axis_combo = self._RECIPE_AXIS_COMBO.get(method)
            axis_idx = None
            if axis_combo is not None:
                axis_idx = getattr(self, axis_combo).findText(str(p["axis"]))
                if axis_idx < 0:
                    return False, "its orientation is not offered here"

            # ...then set: the method first, WITH signals.
            if int(mcombo.currentIndex()) != midx:
                mcombo.setCurrentIndex(midx)
            if method == "contour":
                self.refine_contour_mode_combo.setCurrentIndex(
                    CONTOUR_MODES.index(p["mode"]))
                self.refine_wall_kind_combo.setCurrentIndex(
                    REFINE_WALL_KINDS.index(p["wall_kind"]))
                self.refine_contour_axis_combo.setCurrentIndex(
                    REFINE_CONTOUR_AXES.index(p["axis"]))
                self.refine_contour_slices_combo.setCurrentIndex(
                    SLICE_SAMPLINGS.index(p["sampling"]))
            elif method == "settle":
                self.refine_wall_kind_combo.setCurrentIndex(
                    REFINE_WALL_KINDS.index(p["wall_kind"]))
                self.refine_solver_combo.setCurrentIndex(
                    REFINE_SOLVERS.index(p["solver"]))
            elif method == "sdf":
                self.smooth_preserve_topology_cb.setChecked(
                    bool(p["preserve_topology"]))
            if axis_combo is not None:
                getattr(self, axis_combo).setCurrentIndex(axis_idx)
            for w, v in plan:
                w.setValue(int(v) if isinstance(w, QSpinBox) else float(v))
            if "prep_ticks" in p:
                for key, on in p["prep_ticks"].items():
                    cb = getattr(self, "refine_prep_{}_cb".format(key), None)
                    if cb is not None:
                        cb.setChecked(bool(on))
                if self.refine_prep_chain() != p["prep_chain"]:
                    return False, rr.STAGE_PREP_DIFFERS
        except Exception as exc:  # noqa: BLE001 - never raise into a run
            return False, "the page could not be set ({})".format(exc)
        return True, ""

    def _on_refine_method_changed(self, index: int) -> None:
        """Swap the parameter page, the description and the channel card."""
        if hasattr(self, "_refine_params_stack"):
            self._refine_params_stack.setCurrentIndex(index)
            self._refine_params_stack.updateGeometry()
        if hasattr(self, "_refine_method_desc_label"):
            desc = self._REFINE_METHOD_DESCRIPTIONS.get(index, "")
            self._refine_method_desc_label.setText(desc)
        self._apply_refine_method_visibility()
        cb = getattr(self, "_refine_method_changed_cb", None)
        if cb is not None:
            try:
                cb(int(index))
            except Exception as exc:  # noqa: BLE001
                print("[REFINE] method-changed callback raised: "
                      "{!r}".format(exc))

    # --- the walls tick, the Refine twin of the Reshape one -------

    def set_refine_walls_kind(self, kind=None) -> None:
        """The wall kind the STAGED picture was built with (B3 audit).

        The legend used to read the LIVE combo, so turning the combo after a
        Preview relabelled a picture that had not changed.  ``None`` falls
        back to the combo.
        """
        self._refine_staged_wall_kind = (str(kind) if kind else None)
        cb = getattr(self, "refine_show_walls_cb", None)
        legend = getattr(self, "refine_walls_legend", None)
        if (cb is not None and legend is not None and cb.isEnabled()
                and cb.isChecked()):
            legend.setText(self.refine_walls_legend_text())

    def refine_walls_legend_text(self) -> str:
        """The sentence under the tick, worded from the WALL KIND of the
        staged picture (the combo only when none is recorded)."""
        kind = (getattr(self, "_refine_staged_wall_kind", None)
                or self.refine_wall_kind())
        if kind == "probability":
            return WALLS_LEGEND_PROBABILITY
        if kind == "skeleton":
            return REFINE_WALLS_LEGEND_SPINES
        if kind == "distance":
            return REFINE_WALLS_LEGEND_PINCH
        return WALLS_LEGEND_IMAGE

    def is_refine_show_walls_checked(self) -> bool:
        cb = getattr(self, "refine_show_walls_cb", None)
        return bool(cb is not None and cb.isChecked())

    def set_refine_walls_toggle_enabled(self, enabled: bool,
                                        reason: str = "") -> None:
        """Offer the tick, or say why it is grey -- ON SCREEN.

        Qt shows no tooltip on a disabled widget, so the reason also goes
        into the legend's slot, which is on screen exactly when S5 is.
        """
        cb = getattr(self, "refine_show_walls_cb", None)
        if cb is None:
            return
        cb.setEnabled(bool(enabled))
        cb.setToolTip(
            REFINE_WALLS_TOGGLE_HELP if enabled
            else (str(reason) or WALLS_TOGGLE_DISABLED)
        )
        legend = getattr(self, "refine_walls_legend", None)
        if legend is None:
            return
        if enabled:
            checked = self.is_refine_show_walls_checked()
            legend.setText(self.refine_walls_legend_text() if checked else "")
            legend.setVisible(bool(checked))
        else:
            text = str(reason or "")
            legend.setText(text)
            legend.setVisible(bool(text))

    def clear_refine_walls_toggle(self) -> None:
        """Untick, disable and take the legend away.  SILENT."""
        cb = getattr(self, "refine_show_walls_cb", None)
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
            cb.setEnabled(False)
            cb.setToolTip(WALLS_TOGGLE_DISABLED)
        legend = getattr(self, "refine_walls_legend", None)
        if legend is not None:
            legend.setText("")
            legend.setVisible(False)

    def set_refine_walls_toggle_checked(self, checked: bool) -> None:
        """Put the tick back up SILENTLY after a re-Preview."""
        cb = getattr(self, "refine_show_walls_cb", None)
        if cb is None:
            return
        cb.blockSignals(True)
        cb.setChecked(bool(checked))
        cb.blockSignals(False)
        legend = getattr(self, "refine_walls_legend", None)
        if legend is not None:
            legend.setText(self.refine_walls_legend_text() if checked else "")
            legend.setVisible(bool(checked))

    def set_refine_walls_message(self, text: str) -> None:
        """Put a sentence in the legend's slot -- the budget refusal."""
        legend = getattr(self, "refine_walls_legend", None)
        if legend is None:
            return
        legend.setText(str(text or ""))
        legend.setVisible(bool(text))

    # --- callbacks + state --------------------------------------------------

    def set_smooth_callbacks(
        self,
        *,
        on_preview=None,
        on_accept=None,
        on_reject=None,
        on_prob_toggle=None,
        on_cancel_preview=None,
        on_show_walls_toggled=None,
        on_method_changed=None,
        on_recipes_menu=None,
        on_recipe_run=None,
        on_recipe_load=None,
        on_recipe_save=None,
        on_recipe_delete=None,
    ):
        """Wire the Refine page to external orchestration callbacks.

        Disconnect-before-connect throughout: the dock is REUSED across
        relaunches.  The three new keywords are ADDITIVE -- every existing
        call site stays valid.
        """
        if on_preview and hasattr(self, "smooth_preview_btn"):
            try:
                self.smooth_preview_btn.clicked.disconnect()
            except TypeError:
                pass
            self.smooth_preview_btn.clicked.connect(on_preview)
        if on_accept and hasattr(self, "smooth_accept_btn"):
            try:
                self.smooth_accept_btn.clicked.disconnect()
            except TypeError:
                pass
            self.smooth_accept_btn.clicked.connect(on_accept)
        if on_reject and hasattr(self, "smooth_reject_btn"):
            try:
                self.smooth_reject_btn.clicked.disconnect()
            except TypeError:
                pass
            self.smooth_reject_btn.clicked.connect(on_reject)
        if on_cancel_preview and hasattr(self, "refine_cancel_preview_btn"):
            try:
                self.refine_cancel_preview_btn.clicked.disconnect()
            except TypeError:
                pass
            self.refine_cancel_preview_btn.clicked.connect(on_cancel_preview)
        if on_show_walls_toggled and hasattr(self, "refine_show_walls_cb"):
            try:
                self.refine_show_walls_cb.toggled.disconnect()
            except TypeError:
                pass
            self.refine_show_walls_cb.toggled.connect(on_show_walls_toggled)
        if on_method_changed is not None:
            self._refine_method_changed_cb = on_method_changed
        # Refine recipes: the menu is rebuilt on every
        # opening; disconnect-before-connect, the dock being REUSED across
        # relaunches.  The four item callbacks are read by
        # `populate_refine_recipes_menu` when it builds each action.
        if on_recipes_menu is not None and hasattr(
                self, "refine_recipes_menu"):
            try:
                self.refine_recipes_menu.aboutToShow.disconnect()
            except TypeError:
                pass
            self.refine_recipes_menu.aboutToShow.connect(on_recipes_menu)
        for key, cb in (("run", on_recipe_run), ("load", on_recipe_load),
                        ("save", on_recipe_save),
                        ("delete", on_recipe_delete)):
            if cb is not None:
                self._refine_recipe_cbs[key] = cb
        if on_prob_toggle and hasattr(self, "refine_prob_overlay_cb"):
            try:
                self.refine_prob_overlay_cb.toggled.disconnect()
            except TypeError:
                pass
            self.refine_prob_overlay_cb.toggled.connect(on_prob_toggle)
            # Re-apply overlay when colormap changes while checkbox is on.
            if hasattr(self, "refine_prob_cmap_combo"):
                try:
                    self.refine_prob_cmap_combo.currentIndexChanged.disconnect()
                except TypeError:
                    pass

                def _on_cmap_changed(_idx, cb=self.refine_prob_overlay_cb, fn=on_prob_toggle):
                    if cb.isChecked():
                        fn(True)

                self.refine_prob_cmap_combo.currentIndexChanged.connect(_on_cmap_changed)

    def update_smooth_ui_state(
        self,
        *,
        preview_enabled: bool | None = None,
        accept_enabled: bool | None = None,
        reject_enabled: bool | None = None,
        status_text: str | None = None,
        busy: bool | None = None,
        busy_progress=None,
        busy_text: str | None = None,
        stage_text: str | None = None,
        stop_visible: bool | None = None,
        stop_enabled: bool | None = None,
        preview_text: str | None = None,
        preview_reason: str | None = None,
        accept_text: str | None = None,
        object_text: str | None = None,
        result_note: str | None = None,
        result_warning: str | None = None,
        breadcrumb: str | None = None,
        status_kind: str | None = None,
        rows_enabled: bool | None = None,
    ) -> None:
        """Partial, None-guarded update of every Refine affordance.

        Every parameter defaults to ``None`` and only the ones passed are
        applied, so the 15 existing call sites -- which reach this through
        ``Curation3DAnnotator._update_smooth_ui`` -- all stay valid.
        """
        if preview_enabled is not None and hasattr(self, "smooth_preview_btn"):
            self.smooth_preview_btn.setEnabled(preview_enabled)
        if preview_reason is not None and hasattr(self, "smooth_preview_btn"):
            # A refusal the user can read BEFORE the click.
            self.smooth_preview_btn.setToolTip(
                str(preview_reason) or REFINE_PREVIEW_TIP)
        if accept_enabled is not None and hasattr(self, "smooth_accept_btn"):
            self.smooth_accept_btn.setEnabled(accept_enabled)
        if accept_text is not None and hasattr(self, "smooth_accept_btn"):
            self.smooth_accept_btn.setText(str(accept_text))
        if reject_enabled is not None and hasattr(self, "smooth_reject_btn"):
            self.smooth_reject_btn.setEnabled(reject_enabled)
            # The Refine tab stages exactly when its secondary is live, so
            # the label needs no new keyword to key on.
            self.smooth_reject_btn.setText(
                "Discard" if reject_enabled else "Cancel"
            )
            # ...and so does S4's word: "Preview again"
            # exactly while a result is staged.  Deriving it here is what
            # puts it back after Accept and Discard, which never passed
            # `preview_text` (owner live test 2026-09-23).
            if hasattr(self, "smooth_preview_btn"):
                self.smooth_preview_btn.setText(
                    "Preview again" if reject_enabled else "Preview"
                )
            self._refresh_refine_footer_summary()
        # An explicit word still wins, applied AFTER the derivation.
        if preview_text is not None and hasattr(self, "smooth_preview_btn"):
            self.smooth_preview_btn.setText(str(preview_text))
        if status_text is not None and hasattr(self, "smooth_status_label"):
            self.smooth_status_label.setText(status_text)
        if object_text is not None and hasattr(self, "refine_object_value"):
            self.refine_object_value.setText(str(object_text) or "-")

        # -- the pinned strip ------------------------------------------
        if busy is not None:
            box = getattr(self, "_refine_busy_container", None)
            if box is not None:
                box.setVisible(bool(busy))
            btn = getattr(self, "refine_cancel_preview_btn", None)
            if btn is not None and stop_visible is None:
                btn.setVisible(bool(busy))
                btn.setEnabled(bool(busy))
        if busy_progress is not None:
            bar = getattr(self, "refine_busy_bar", None)
            if bar is not None:
                try:
                    i, n = int(busy_progress[0]), int(busy_progress[1])
                except Exception:
                    i, n = 0, 0
                if n > 0:
                    bar.setRange(0, n)
                    bar.setValue(i)
                else:
                    bar.setRange(0, 0)
        if stop_visible is not None:
            btn = getattr(self, "refine_cancel_preview_btn", None)
            if btn is not None:
                btn.setVisible(bool(stop_visible))
        if stop_enabled is not None:
            btn = getattr(self, "refine_cancel_preview_btn", None)
            if btn is not None:
                btn.setEnabled(bool(stop_enabled))
        # `busy_text` and `stage_text` are the SAME slot, declared under
        # both names because  names `busy_text` (the Reshape
        # keyword) and the page calls the thing a stage line.  `stage_text`
        # wins when both are passed.
        _stage = stage_text if stage_text is not None else busy_text
        if _stage is not None:
            lb = getattr(self, "refine_stage_label", None)
            if lb is not None:
                # ONE line: the whole sentence rides in the tooltip, so a
                # long stage line cannot make the pinned strip grow -- and
                # the visible text is right-elided to the label's width.
                self._refine_stage_full_text = str(_stage)
                lb.setToolTip(str(_stage))
                lb.setVisible(bool(_stage))
                self._apply_refine_stage_elision()

        # -- S5 --------------------------------------------------------
        if breadcrumb is not None:
            lb = getattr(self, "refine_breadcrumb", None)
            if lb is not None:
                lb.setText(str(breadcrumb))
                lb.setVisible(bool(breadcrumb))
        if result_note is not None:
            lb = getattr(self, "refine_result_note", None)
            if lb is not None:
                lb.setText(str(result_note))
                lb.setVisible(bool(result_note))
        if result_warning is not None:
            lb = getattr(self, "refine_result_warning", None)
            if lb is not None:
                lb.setText(str(result_warning))
                lb.setVisible(bool(result_warning))
        if (result_note is not None or result_warning is not None
                or breadcrumb is not None):
            box = getattr(self, "_refine_result_container", None)
            if box is not None:
                # `isHidden()`, NOT `isVisible()`: this page is behind
                # another tab most of the time, so every child of it
                # answers False to `isVisible()` and the card would never
                # open.  `isHidden()` asks the question actually being
                # decided -- has this row been taken off the page.
                box.setVisible(any(
                    (getattr(self, attr, None) is not None
                     and not getattr(self, attr).isHidden())
                    for attr in ("refine_result_note",
                                 "refine_result_warning",
                                 "refine_breadcrumb")
                ))

        # -- S3 is DISABLED, not hidden, while a worker runs, so the page
        #    does not jump under the cursor.
        if rows_enabled is not None:
            for attr in ("_refine_method_card", "_refine_target_card",
                         "_refine_advanced_container"):
                w = getattr(self, attr, None)
                if w is not None:
                    w.setEnabled(bool(rows_enabled))

        if status_kind is not None:
            self._set_status_dot("_refine_status_dot", str(status_kind))
        elif reject_enabled is not None or accept_enabled is not None:
            self._set_status_dot(
                "_refine_status_dot",
                "review" if self.smooth_reject_btn.isEnabled() else "idle",
            )

    def _build_otsu_section(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        desc = QLabel(
            "Automatic intensity thresholding on the current slice. "
            "Generates a binary mask from image intensities."
        )
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Layer selectors
        form = QFormLayout()
        form.setVerticalSpacing(2)
        self.otsu_image_combo = QComboBox()
        self.otsu_target_combo = QComboBox()
        form.addRow("Image:", self.otsu_image_combo)
        form.addRow("Target:", self.otsu_target_combo)
        layout.addLayout(form)

        # Method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.otsu_method_combo = QComboBox()
        self.otsu_method_combo.addItems(["Otsu", "Multi-Otsu", "Li", "Yen", "Triangle"])
        method_row.addWidget(self.otsu_method_combo, 1)
        layout.addLayout(method_row)

        # Action buttons
        self.otsu_preview_btn = QPushButton("Preview")
        self.otsu_preview_btn.setStyleSheet(_REFINEMENT_PREVIEW_STYLE)
        self.otsu_accept_btn = QPushButton("Accept")
        self.otsu_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.otsu_reject_btn = QPushButton("Reject")
        self.otsu_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)

        layout.addWidget(self.otsu_preview_btn)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.otsu_accept_btn)
        btn_row.addWidget(self.otsu_reject_btn)
        layout.addLayout(btn_row)

        notice = QLabel("Backend not connected — UI placeholder.")
        notice.setStyleSheet("color: #777; font-style: italic;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        return group

    # --- Watershed Split — HITL 2D -----------------------------------------

    def _build_watershed_section(self) -> QWidget:
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        desc = QLabel(
            "Split touching or merged labels using marker-controlled "
            "watershed on distance transform or intensity."
        )
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Target layer
        form = QFormLayout()
        form.setVerticalSpacing(2)
        self.ws_image_combo = QComboBox()
        self.ws_target_combo = QComboBox()
        form.addRow("Image:", self.ws_image_combo)
        form.addRow("Labels:", self.ws_target_combo)
        layout.addLayout(form)

        # Seed mode
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seeds:"))
        self.ws_seed_combo = QComboBox()
        self.ws_seed_combo.addItems(["Distance Transform", "Manual Markers", "Local Maxima"])
        seed_row.addWidget(self.ws_seed_combo, 1)
        layout.addLayout(seed_row)

        # Action buttons
        self.ws_run_btn = QPushButton("Run Watershed")
        self.ws_accept_btn = QPushButton("Accept")
        self.ws_accept_btn.setStyleSheet(_REFINEMENT_ACCEPT_STYLE)
        self.ws_reject_btn = QPushButton("Reject")
        self.ws_reject_btn.setStyleSheet(_REFINEMENT_REJECT_STYLE)

        layout.addWidget(self.ws_run_btn)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.ws_accept_btn)
        btn_row.addWidget(self.ws_reject_btn)
        layout.addLayout(btn_row)

        notice = QLabel("Backend not connected — UI placeholder.")
        notice.setStyleSheet("color: #777; font-style: italic;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        return group

