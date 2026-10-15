"""Step 4's stack rows, painted (Appendix A ``.row`` /
``.nrow``) -- shared with Step 3: moved here from ``segmentation/``, whose
``_stack_row_delegate`` is now an alias of THIS module object.

``StackRowDelegate`` paints the three ``QListWidget``s of Step 4 from the
immutable row record each item carries in ``Qt.UserRole + 2`` (built by
``_step4_rows.build_row``):

* ``"wide"`` (① Session): tick · 32 px thumbnail · name and a right-aligned
  mono status word · a dim line 2 ``{axes} · {c} ch · {spacing}`` (the word
  "voxels" amber) · amber warning lines · an eye and a ``⋯`` menu.
* ``"narrow"`` (② Segment): tick · status icon · the mono name / the status
  word and dots on the left with the axis and an eye on the right / the dim
  shape line / chips and amber warnings.
* ``"measure"`` (③): the narrow row with no tick.

The delegate holds no state.  The tick, the eye and the menu are
hit-rectangles handled in :meth:`editorEvent`; the tick writes the item's
check state (so the list's ``itemChanged`` fires), the eye and the menu are
signals.  Help on the painted parts comes from :meth:`helpEvent` with the
lines ``row_tick`` / ``row_chip`` / ``row_eye`` (and ``mt_row_click`` for ③)
of the ``help_lines`` mapping handed to the constructor, and the wide row's
line 2 from its ``wide_line2`` template (``{axes}``, ``{c}``,
``{spacing}``): the module imports nothing from a step package, so Step 4
passes ``STEP4_HELP_LINES`` / ``ROW_WIDE_LINE2``.  Every size depends on the
list's width, so ``StackListWidget`` re-lays its items out on a resize.

Step 3's additive knobs (brick V2), each
defaulting to Step 4's behaviour: ``show_tick=False`` drops the tick (Session
and Curation's rows), a wide row built with ``show_eye=False`` has no eye and
a menu glyph only when its record asks for one (``"menu"``), a record's
``"line2"`` text replaces the template's head (an empty one takes no
height), an item's ``ROLE_THUMB_TIP`` is the picture's own tooltip, and
``keep_aspect=True`` fits a picture that is not square inside its square
instead of stretching it.

Step 3's R6 knobs, defaulting to
Step 4's behaviour: ``show_info=True`` draws an ⓘ in a narrow row's eye slot
when its record asks for one (``"info"``) and emits ``info_clicked(name)``
on a click; a record's ``"chips_top"`` are chips drawn right under the
shape line (the "made by" chip), help key ``row_made_by``; the ⓘ's help key
is ``row_info``.  Step 4 sets neither, so its rows paint as before.

Step 3's R7 knob, defaulting to Step 4's behaviour: a wide record's
``"line3"`` is a mono line under line 2 (Session's voxel size), in the
palette's ``"warn"`` colour when ``"line3_warn"`` is set (the amber
"voxels").  Step 4's records never carry it.

Step 3's A2 keys, wide rows only,
which Step 4's records never carry: ``"row_action"`` = ``(key, word)`` is a
small button drawn after line 3's text (Session's inline "Set spacing…"); a
click emits ``action_clicked(name, key)``, help key ``row_action``, dimmed
under the lock like the menu.  ``"tier_counts"`` = ``((word, n, (r, g,
b)), ...)`` is a line under the last line -- a dot in the tier's colour,
the number, the word -- that drops the tiers which do not fit (``…``).

Step 2's keys (brick L), every one default-off so a
record without them paints exactly as before (golden-tested, byte for
byte): ``"thumb_px"`` = this row's picture edge; ``"no_icon"`` = a narrow
row without its status-icon slot; ``"line2_plain"`` = a narrow row's line 2
is one dim text (no status word, no axis); ``"status_counts"`` = ``((icon,
colour, n, word), ...)`` the icon-counts line (wide: the non-zero ones with
number and word; narrow: every slot, icon and number only, a zero slot
faint), ``"dim_slots"`` dims every slot and blanks the numbers,
``"grey_word"`` is a grey word in the counts line's place, ``"chips_inline"``
= ``((text, kind), ...)`` chips that follow the counts on the same flow,
``"strip"`` = ``((fraction, colour), ...)`` a thin proportion strip across
the foot of the text column.  Polish additions, default-off: a narrow row's
``"shape_variants"`` makes the shape line ONE line fitted from shorter
spellings (instead of a wrapped one), and a list with
``empty_one_line_when_short`` shows its whole empty message when a tighter
rectangle holds it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from qtpy.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QStyle,
    QStyledItemDelegate,
    QToolTip,
)

from ._stack_thumbs import THUMB_NARROW_PX, THUMB_WIDE_PX
from .theme import STEP4_ACCENT

ROLE_NAME = Qt.UserRole
ROLE_STATUS = Qt.UserRole + 1
ROLE_RECORD = Qt.UserRole + 2
ROLE_THUMB = Qt.UserRole + 3
#: the picture's own tooltip (Step 3, V2); Step 4 sets none
ROLE_THUMB_TIP = Qt.UserRole + 4

MODE_WIDE = "wide"
MODE_NARROW = "narrow"
MODE_MEASURE = "measure"

# Appendix A skin (the v7 mockup's .row / .nrow / .chip / .chk / .thumb)
_SEL_BG = QColor(106, 191, 105, 64)
_NAME = QColor("#dddddd")
_NAME_OFF = QColor("#777777")
_DIM = QColor("#999999")
_AMBER = QColor("#d4a843")
_CHIP_TEXT = QColor("#aaaaaa")
_CHIP_BORDER = QColor(255, 255, 255, 36)
_CHIP_WARN_BORDER = QColor(212, 168, 67, 140)
#: the unticked box's 2 px border (the rows mockup v3's ``.tick``)
_CHK_BORDER = QColor("#5a5e69")
_THUMB_BG = QColor("#111111")
_THUMB_IN = QColor("#2c2c2c")
#: R1-5: a disabled CHECKED box keeps its tick but in a dimmed blue, so a
#: greyed setting never reads as live (napari's checked blue is #4560c4; this is ~45 % of it, matching
#: the delegate's dimmed tick).  Added to Step 4's panel and Export window.
CHECK_DISABLED_BLUE = "#34427a"
CHECK_DISABLED_STYLE = (
    "QCheckBox::indicator:checked:disabled { background-color: "
    + CHECK_DISABLED_BLUE + "; }"
    "QListWidget::indicator:checked:disabled { background-color: "
    + CHECK_DISABLED_BLUE + "; }")
_ICON_DIM = "#999999"

#: the delegate's colours (brick V1).
#: Step 4's values are the defaults, so a delegate built without
#: ``palette=`` paints as Step 4 does.  A3: the tick is ONE rule for both
#: steps -- a 2 px grey border unticked, the list's ``accent`` fill plus a
#: 2 px halo at 35 % ticked -- the accent a constructor keyword (Step 3
#: passes PANEL_GOLD, Step 4 defaults to STEP4_ACCENT); ``chk_mark`` is the
#: tick glyph's colour (Step 3: dark on amber).
DEFAULT_PALETTE = {
    "sel_bg": _SEL_BG,
    "accent": QColor(STEP4_ACCENT),
    "chk_mark": QColor("#ffffff"),
    "warn": _AMBER,
}

#: the loaded-row indicator (A3 fold): bar width and glow strength
LOADED_BAR_PX = 2
LOADED_GLOW_ALPHA = 0.10
TICK = 16
#: the tick's halo (rows mockup v3: ``0 0 0 2px`` of the accent at 35 %)
TICK_HALO = 2
TICK_HALO_ALPHA = 0.35
ICON = 14
#: Step 2's proportion strip and its gap (``"strip"`` key)
STRIP_PX = 3
STRIP_GAP = 4
PAD_X_WIDE = 6
PAD_X_NARROW = 5
PAD_Y = 4
GAP = 1

_MONO_FAMILIES = ["Consolas", "Cascadia Mono", "Courier New"]


def thumb_px(mode: str) -> int:
    """The picture's edge in *mode*: ``THUMB_WIDE_PX`` on a wide row,
    ``THUMB_NARROW_PX`` on a narrow / measure row (A3)."""
    return THUMB_WIDE_PX if mode == MODE_WIDE else THUMB_NARROW_PX


#: A list that scrolls inside itself shows this many picture rows
#: (so the list scrolls inside itself)
LIST_CAP_ROWS = 4
#: the room a list's frame and its own padding take around its rows
LIST_CAP_CHROME = 6


def list_cap_px(mode: str = MODE_WIDE, rows: int = LIST_CAP_ROWS) -> int:
    """The height of a list that shows ``rows`` rows of *mode*'s picture
    (a row is at least the picture plus its padding): the cap of Session's
    list in Step 3 and Step 4, derived from the picture constants."""
    return int(rows) * (thumb_px(mode) + 2 * PAD_Y) + LIST_CAP_CHROME


def _shifted(value, dy: int):
    """*value* (a layout entry) moved down by *dy*: a rectangle translated,
    a dict / list / tuple of them rebuilt, anything else as it was."""
    if isinstance(value, QRect):
        return value.translated(0, dy)
    if isinstance(value, dict):
        return {k: _shifted(v, dy) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_shifted(v, dy) for v in value)
    return value


def _vox_numbers(text) -> str:
    """A voxel-size line without its source: "0.52 × 0.52 × 0.52 µm (from
    Step 1)" -> "0.52 × 0.52 × 0.52 µm" (the source rides the tooltip)."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(text or ""))


def _is_checked(value) -> bool:
    """True when ``Qt.CheckStateRole`` data means Checked.  Under PyQt6 the
    role can hold a bare int (``setCheckState`` and Qt's own Space toggle
    store ``int(state)``) that never equals the ``Qt.CheckState`` enum, so
    compare the values; ``None`` is unchecked."""
    if value is None:
        return False
    return getattr(value, "value", value) == getattr(Qt.Checked, "value",
                                                     Qt.Checked)


def _font(base: QFont, px: int, mono: bool = False, bold: bool = False):
    f = QFont(base)
    if mono:
        f.setFamilies(_MONO_FAMILIES) if hasattr(f, "setFamilies") else \
            f.setFamily(_MONO_FAMILIES[0])
        f.setStyleHint(QFont.Monospace)
    f.setPixelSize(px)
    f.setBold(bold)
    return f


def _icon_pixmap(name: str, colour: str, size: int = ICON):
    try:
        import qtawesome as qta
        return qta.icon(name, color=colour).pixmap(size, size)
    except Exception:      # noqa: BLE001 -- cosmetic
        return None


def _elide_tail(fm: QFontMetrics, text: str, width: int,
                tail: str = "") -> str:
    """*text* elided at the TAIL (the head stays readable) to *width* px.
    A *tail* it ends with (the amber " -- not the active model") is shown
    only when the whole chip fits; otherwise it is dropped from the picture
    first (the chip's amber border and its tooltip say it) and the name is
    elided on the right."""
    if fm.horizontalAdvance(text) <= width:
        return text
    if tail and text.endswith(tail):
        text = text[:-len(tail)]
        if fm.horizontalAdvance(text) <= width:
            return text
    return fm.elidedText(text, Qt.ElideRight, width)


def _elide_keep_tail(fm: QFontMetrics, text: str, width: int,
                     keep: int) -> str:
    """*text* elided in the MIDDLE but keeping its last *keep* characters
    whole (``WholeScale_patch_8`` -> ``Whole...patch_8``): the part that
    tells rows apart is the tail.  Falls back to the plain middle elision
    when even the tail cannot fit."""
    if fm.horizontalAdvance(text) <= width:
        return text
    keep = max(1, min(int(keep), len(text) - 1))
    tail = text[-keep:]
    room = width - fm.horizontalAdvance("…" + tail)
    if room <= 0:
        return fm.elidedText(text, Qt.ElideMiddle, width)
    head = fm.elidedText(text[:-keep], Qt.ElideRight, room + fm.horizontalAdvance(
        "…"))
    if not head.endswith("…"):          # the whole head fitted
        head += "…"
    return head + tail


def _fit_variant(fm: QFontMetrics, variants, width: int) -> str:
    """The first of *variants* (longest first) that fits *width*; the last
    one, elided, when none does."""
    variants = [str(v) for v in variants]
    for v in variants:
        if fm.horizontalAdvance(v) <= width:
            return v
    return fm.elidedText(variants[-1], Qt.ElideRight, width)         if variants else ""


def _wrap_height(fm: QFontMetrics, text: str, width: int) -> int:
    if not text:
        return 0
    r = fm.boundingRect(QRect(0, 0, max(width, 10), 10000),
                        int(Qt.TextWordWrap | Qt.AlignLeft), text)
    return r.height()


class StackRowDelegate(QStyledItemDelegate):
    """Paints one Step 4 stack row (``mode``: wide / narrow / measure)."""

    eye_clicked = Signal(str)
    menu_requested = Signal(str, QPoint)
    #: the ⓘ of a row (Step 3, R6; ``show_info=True`` only)
    info_clicked = Signal(str)
    #: a wide row's inline button (Step 3, A2): name, the record's key
    action_clicked = Signal(str, str)

    def __init__(self, mode: str = MODE_NARROW, parent=None, *,
                 help_lines=None, wide_line2=None, palette=None,
                 show_eye: bool = True, lock_property: str = "step4Locked",
                 eye_lock_property: str = "step4EyeLocked",
                 dim_when_locked: bool = False, show_tick: bool = True,
                 keep_aspect: bool = False, show_info: bool = False,
                 show_thumb: bool = False, accent=None,
                 wrap_tiers: bool = False):
        super().__init__(parent)
        self._mode = mode
        #: a narrow row whose record asks for it (``"info"``) draws an ⓘ in
        #: the eye's slot (Step 3's model details, R6); Step 4: off
        self._show_info = bool(show_info)
        #: the row's tick (Step 3's Session and Curation rows: off -- the
        #: patches are ticked in Predict + Reconstruct)
        self._show_tick = bool(show_tick)
        #: a NARROW / measure row draws the picture too (Step 3's Predict +
        #: Reconstruct list, A1; Step 4's ② and ③, A3); wide rows always do
        self._show_thumb = bool(show_thumb)
        #: a WIDE row's tier line wraps (words while they fit, dot and number
        #: only when a single tier's words do not) instead of dropping the
        #: tiers that do not fit with "…" -- Curation's box list (A3 fold)
        self._wrap_tiers = bool(wrap_tiers)
        #: the picture keeps its aspect, centred in its square (Step 3: a
        #: 240 x 195 slice is not stretched); Step 4 fills the square
        self._keep_aspect = bool(keep_aspect)
        #: colours over Step 4's (``DEFAULT_PALETTE``); Step 3 passes gold
        #: ticks, an amber selection tint and PANEL_WARN's text colour
        self._palette = dict(DEFAULT_PALETTE)
        if palette:
            self._palette.update(palette)
        if accent is not None:
            self._palette["accent"] = QColor(accent)
        #: the eye (Step 3: off -- View is the foot's button); a wide row
        #: without it shows the menu glyph only when its record asks
        self._show_eye = bool(show_eye)
        #: the list properties the lock sets (Step 3: ``step3Locked``)
        self._lock_property = lock_property
        self._eye_lock_property = eye_lock_property
        #: the whole row dims while the lock property is set (Step 3: the
        #: mockup's locked list; Step 4 dims only the names, D3)
        self._dim_when_locked = bool(dim_when_locked)
        #: key -> help line of the painted parts (Step 4: STEP4_HELP_LINES)
        self._help_lines = help_lines if help_lines is not None else {}
        #: the wide row's line 2 (Step 4: ROW_WIDE_LINE2)
        self._wide_line2 = wide_line2 or ""

    @property
    def mode(self) -> str:
        return self._mode

    # -- geometry ---------------------------------------------------------
    def _fonts(self, base: QFont):
        return {
            "name_wide": _font(base, 11),
            "name_mono": _font(base, 11, mono=True),
            "word": _font(base, 11, mono=True),
            "word_small": _font(base, 10, mono=True),
            "dim": _font(base, 10),
            "chip": _font(base, 10),
            "num": _font(base, 10, mono=True, bold=True),
        }

    def layout(self, rect: QRect, rec: Dict[str, Any], base: QFont) -> dict:
        """Every rectangle of one row, and its total height.  ``rect`` is
        the row's rectangle (only its x, y and width are read)."""
        f = self._fonts(base)
        g: Dict[str, Any] = {"chips": [], "warns": [], "lines": {}}
        x0, y0, w = rect.x(), rect.y(), rect.width()
        wide = self._mode == MODE_WIDE
        pad_x = PAD_X_WIDE if wide else PAD_X_NARROW
        y = y0 + PAD_Y
        x = x0 + pad_x
        tpx = int(rec.get("thumb_px") or thumb_px(self._mode))
        if self._mode in (MODE_WIDE, MODE_NARROW) and self._show_tick:
            # A3: centred vertically once the row's height is known (below)
            g["tick"] = QRect(x, y + 1, TICK, TICK)
            x += TICK + (8 if wide else 6)
        if wide:
            g["thumb"] = QRect(x, y, tpx, tpx)
            x += tpx + 8
            right = x0 + w - pad_x
            if self._show_eye and rec.get("eye") is not False:
                g["menu"] = QRect(right - ICON - 2, y + 1, ICON + 2, ICON + 2)
                g["eye"] = QRect(right - 2 * ICON - 12, y + 1, ICON + 2,
                                 ICON + 2)
                mid_r = g["eye"].left() - 8
            elif self._show_eye:
                # Step 3 (A2 fold): a row whose record says ``"eye": False``
                # (nothing to view yet) keeps its menu, not the eye
                g["menu"] = QRect(right - ICON - 2, y + 1, ICON + 2, ICON + 2)
                mid_r = g["menu"].left() - 8
            elif rec.get("menu"):
                g["menu"] = QRect(right - ICON - 2, y + 1, ICON + 2, ICON + 2)
                mid_r = g["menu"].left() - 8
            else:
                mid_r = right
        else:
            if self._mode in (MODE_NARROW, MODE_MEASURE) and self._show_thumb:
                g["thumb"] = QRect(x, y, tpx, tpx)
                x += tpx + 6
            if not rec.get("no_icon"):
                g["icon"] = QRect(x, y + 2, ICON, ICON)
                x += ICON + 6
            mid_r = x0 + w - pad_x
        mid_w = max(mid_r - x, 40)
        g["mid"] = QRect(x, y, mid_w, 0)
        name_fm = QFontMetrics(f["name_wide"] if wide else f["name_mono"])
        dim_fm = QFontMetrics(f["dim"])
        line1_h = name_fm.height()
        g["lines"]["l1"] = QRect(x, y, mid_w, line1_h)
        y += line1_h + GAP
        if wide:
            if rec.get("line2") == "":
                # Step 3: nothing to say on line 2 -- no blank line before
                # an amber one (Step 4 never sets "line2")
                g["lines"]["l2"] = QRect(x, y, mid_w, 0)
            else:
                h2 = dim_fm.height()
                g["lines"]["l2"] = QRect(x, y, mid_w, h2)
                y += h2 + GAP
            if rec.get("line3"):
                # Step 3 (R7): the voxel-size line under line 2, mono
                sfm = QFontMetrics(f["word_small"])
                h3 = sfm.height()
                act = rec.get("row_action")
                if act:
                    # Step 3 (A2): the inline button after the text
                    afm = QFontMetrics(f["chip"])
                    h3 = max(h3, afm.height() + 4)
                    tw = sfm.horizontalAdvance(str(rec["line3"]))
                    bx = x + tw + 8
                    bw = min(afm.horizontalAdvance(str(act[1])) + 14,
                             max(x + mid_w - bx, 0))
                    if bw > 20:
                        g["action"] = QRect(bx, y, bw, h3)
                g["lines"]["l3w"] = QRect(x, y, mid_w, h3)
                y += h3 + GAP
                tail3 = rec.get("line3_tail")
                if tail3 and sfm.horizontalAdvance(
                        str(rec["line3"]) + str(tail3)) > mid_w:
                    # S1 (audit M10): a tail that would be cut gets its own
                    # line instead of misleading half a number
                    g["lines"]["l3tail"] = QRect(x, y, mid_w, h3)
                    y += h3 + GAP
            tiers = rec.get("tier_counts") or ()
            if tiers:
                # Step 3 (A2): dot · number · word per tier, what fits
                y = self._layout_tiers(g, tiers, x, y, mid_w, f,
                                       wrap=self._wrap_tiers)
            if rec.get("status_counts") is not None or rec.get("grey_word"):
                y = self._layout_counts(g, rec, x, y, mid_w, f, words=True)
            warns = rec.get("warns_wide") or ()
        else:
            word_fm = QFontMetrics(f["word_small"])
            plain2 = rec.get("line2_plain")
            h2 = (dim_fm.height() if plain2 is not None
                  else max(word_fm.height(), ICON))
            g["lines"]["l2"] = QRect(x, y, mid_w, h2)
            ax_w = QFontMetrics(f["word_small"]).horizontalAdvance(
                rec.get("axes") or "")
            eye_x = x + mid_w - ICON
            if plain2 is not None:
                # Step 2 (L): one dim text -- no status word, no axis
                pass
            elif self._mode == MODE_NARROW and self._show_eye:
                g["eye"] = QRect(eye_x - 1, y, ICON + 2, ICON + 2)
                g["axis"] = QRect(eye_x - 6 - ax_w, y, ax_w, h2)
            elif (self._mode == MODE_NARROW and self._show_info
                  and rec.get("info") and not rec.get("menu")):
                # Step 3 (R6): the ⓘ takes the eye's slot
                g["info"] = QRect(eye_x - 1, y, ICON + 2, ICON + 2)
                g["axis"] = QRect(eye_x - 6 - ax_w, y, ax_w, h2)
            elif self._mode == MODE_NARROW and rec.get("menu"):
                # a row with a menu and no eye: the menu glyph takes its place
                g["menu"] = QRect(eye_x - 1, y, ICON + 2, ICON + 2)
                g["axis"] = QRect(eye_x - 6 - ax_w, y, ax_w, h2)
            else:
                g["axis"] = QRect(x + mid_w - ax_w, y, ax_w, h2)
            y += h2 + GAP
            shape = rec.get("shape_text") or ""
            if shape:
                # Step 2 (polish): ``"shape_variants"`` = shorter spellings of
                # the shape line, longest first -> ONE line, never a wrap
                h3 = (dim_fm.height() if rec.get("shape_variants")
                      else _wrap_height(dim_fm, shape, mid_w))
                g["lines"]["l3"] = QRect(x, y, mid_w, h3)
                y += h3 + GAP
            if rec.get("vox_text"):
                # A3: the voxel-size line under the shape line, mono, amber
                # ("voxels") when the record says so; no inline button
                # (audit H1) the NUMBERS only, wrapped like the shape line --
                # the source "(from Step 1)" is the row's tooltip; a narrow
                # column cut the whole line to "0.52 × 0.52 ×…" before
                shown = _vox_numbers(rec["vox_text"])
                g["vox_shown"] = shown
                h4 = max(_wrap_height(QFontMetrics(f["word_small"]), shown,
                                      mid_w),
                         QFontMetrics(f["word_small"]).height())
                g["lines"]["l3n"] = QRect(x, y, mid_w, h4)
                y += h4 + GAP
            top = rec.get("chips_top") or ()
            if top:
                # Step 3 (R6): the "made by" chip, right under the shape line
                g["chips_top"] = []
                y = self._layout_chips(g["chips_top"], top, x, y, mid_w,
                                       QFontMetrics(f["chip"]),
                                       elide=bool(rec.get(
                                           "chips_top_elide")),
                                       tail=str(rec.get(
                                           "chips_top_tail") or ""))
            notes = rec.get("notes_narrow") or ()
            if notes:
                g["notes"] = []
                for text in notes:
                    h = _wrap_height(dim_fm, text, mid_w)
                    g["notes"].append((QRect(x, y, mid_w, h), text))
                    y += h + GAP
            warns = rec.get("warns_narrow") or ()
        for text in warns:
            h = _wrap_height(dim_fm, text, mid_w)
            g["warns"].append((QRect(x, y, mid_w, h), text))
            y += h + GAP
        if not wide:
            y = self._layout_chips(g["chips"], rec.get("chips") or (), x, y,
                                   mid_w, QFontMetrics(f["chip"]),
                                   elide=bool(rec.get("chips_elide")))
            tiers = rec.get("tier_counts") or ()
            if tiers:
                # A3: the protected rows' tier line under the chip -- dot
                # and number (the word is the tooltip's), wrapped
                words = bool(rec.get("tier_words"))
                y = self._layout_tiers(g, tiers, x, y, mid_w, f,
                                       compact=not words, wrap=words)
            if rec.get("status_counts") is not None or rec.get("grey_word"):
                y = self._layout_counts(g, rec, x, y, mid_w, f, words=False)
        if rec.get("strip"):
            g["strip"] = QRect(x, y + 1, mid_w, STRIP_PX)
            y += STRIP_PX + STRIP_GAP
        bottom = y + PAD_Y - GAP
        height = bottom - y0
        if "thumb" in g:
            floor = PAD_Y + tpx + PAD_Y
            if height < floor:
                # A3: the text block is centred on the picture
                dy = (floor - height) // 2
                for key in list(g):
                    if key not in ("thumb", "tick", "height"):
                        g[key] = _shifted(g[key], dy)
                height = floor
            g["thumb"] = QRect(g["thumb"].x(), y0 + (height - tpx) // 2,
                               tpx, tpx)
        if "tick" in g:
            # A3: centred vertically on the row, in both modes
            g["tick"] = QRect(g["tick"].x(), y0 + (height - TICK) // 2,
                              TICK, TICK)
        g["height"] = height
        return g

    @staticmethod
    def _layout_tiers(g: dict, tiers, x: int, y: int, mid_w: int,
                      f: dict, compact: bool = False,
                      wrap: bool = False) -> int:
        """Step 3's tier line (A2) into *g*: ``lines["tiers"]``,
        ``tier_items`` (the hit box and word of each tier drawn) and
        ``tier_draw`` (dot, number and word rectangles); a tier that does
        not fit is dropped with every one after it (``tiers_elided``).  The
        y under the line.

        ``compact`` (A3, the narrow rows): each tier is a dot and its
        number only (the word is in the tooltip), and the line WRAPS onto
        as many rows as it needs instead of dropping a tier.  ``wrap`` keeps
        the words (dot · number · word, as on the wide rows) but wraps the
        same way -- a narrow row whose record says ``tier_words`` (Step 4's
        ③); P+R's protected rows stay compact."""
        nfm, dfm = QFontMetrics(f["num"]), QFontMetrics(f["dim"])
        h = max(nfm.height(), dfm.height())
        line = QRect(x, y, mid_w, h)
        g["lines"]["tiers"] = line
        g["tier_items"], g["tier_draw"], g["tiers_elided"] = [], [], False
        if wrap and not compact:
            # words while a single tier's words fit; else dot + number
            widest = max((12 + nfm.horizontalAdvance(f"{int(n):,}") + 4
                          + dfm.horizontalAdvance(w)
                          for w, n, _c in tiers), default=0)
            compact = widest > mid_w
        if compact or wrap:
            cx, cy = x, y
            for word, n, rgb in tiers:
                num = f"{int(n):,}"
                nw = nfm.horizontalAdvance(num)
                ww = 0 if compact else dfm.horizontalAdvance(word)
                w = 8 + 4 + nw + (0 if compact else 4 + ww)
                if cx > x and cx + w > x + mid_w:
                    cx, cy = x, cy + h + 1
                dot = QRect(cx, cy + (h - 8) // 2, 8, 8)
                nr = QRect(cx + 12, cy, nw + 1, h)
                wr = (QRect(0, 0, 0, 0) if compact
                      else QRect(nr.right() + 4, cy, ww + 1, h))
                g["tier_items"].append((QRect(cx, cy, w, h), word))
                g["tier_draw"].append((dot, nr, wr, num,
                                       "" if compact else word,
                                       tuple(rgb)))
                cx += w + 10
            g["lines"]["tiers"] = QRect(x, y, mid_w, cy + h - y)
            return cy + h + GAP
        ell = dfm.horizontalAdvance("…") + 4
        cx = x
        for word, n, rgb in tiers:
            num = f"{int(n):,}"
            nw, ww = nfm.horizontalAdvance(num), dfm.horizontalAdvance(word)
            w = 8 + 4 + nw + 4 + ww
            room = line.right() - cx + 1
            if w > room:
                g["tiers_elided"] = True
                break
            dot = QRect(cx, y + (h - 8) // 2, 8, 8)
            nr = QRect(cx + 12, y, nw + 1, h)
            wr = QRect(nr.right() + 4, y, ww + 1, h)
            g["tier_items"].append((QRect(cx, y, w, h), word))
            g["tier_draw"].append((dot, nr, wr, num, word, tuple(rgb)))
            cx += w + 10
        if g["tiers_elided"]:
            # the "…" always shows: drop more tiers until it fits
            while g["tier_items"] and cx + ell > line.right() + 1:
                g["tier_items"].pop()
                cx = g["tier_draw"].pop()[0].left()
            if cx + ell <= line.right() + 1:
                g["tier_ellipsis"] = QRect(cx, y, ell, h)
        return y + h + GAP

    @staticmethod
    def _layout_counts(g: dict, rec: dict, x: int, y: int, mid_w: int,
                       f: dict, *, words: bool) -> int:
        """Step 2's icon-counts line (L) into *g*: ``counts`` (icon, number
        and word rectangles of each slot drawn), ``count_items`` (the hit
        box and tooltip of each), ``count_chips`` and ``lines["counts"]``;
        a flow that wraps.  Wide (*words*): only the non-zero statuses, icon
        + number + word, then the inline chips.  Narrow: every slot, icon +
        number, a zero slot faint.  ``grey_word`` replaces the whole line.
        The y under the line."""
        nfm, dfm = QFontMetrics(f["num"]), QFontMetrics(f["dim"])
        cfm = QFontMetrics(f["chip"])
        h = max(nfm.height(), dfm.height(), ICON)
        if rec.get("chips_inline"):
            h = max(h, cfm.height() + 4)       # a chip's own padding
        g["counts"], g["count_items"], g["count_chips"] = [], [], []
        grey = rec.get("grey_word")
        if grey:
            g["lines"]["counts"] = QRect(x, y, mid_w, h)
            g["grey_word"] = (QRect(x, y, mid_w, h), str(grey))
            return y + h + GAP
        dim_all = bool(rec.get("dim_slots"))
        slots = tuple(rec.get("status_counts") or ())
        # Step 2 (fix 2): a narrow row whose five slots would not fit on
        # one line shows the NON-ZERO statuses only (all dimmed slots,
        # icon only, when there is none); a wide row keeps the dimmed slots
        compact = False
        if rec.get("counts_compact") and not words:
            full = sum(ICON + 3 + max(nfm.horizontalAdvance(
                "" if (int(s[2]) <= 0 or dim_all) else f"{int(s[2]):,}"),
                nfm.horizontalAdvance("00")) + 8 for s in slots)
            compact = full > mid_w
        cx, cy = x, y
        for icon, colour, n, word in slots:
            n = int(n)
            if words and n <= 0 and not dim_all:
                continue
            if compact and n <= 0 and not dim_all:
                continue
            num = "" if (n <= 0 or dim_all) else f"{n:,}"
            nw = max(nfm.horizontalAdvance(num),
                     0 if (words or compact)
                     else nfm.horizontalAdvance("00"))
            ww = dfm.horizontalAdvance(str(word)) if words else 0
            w = ICON + 3 + nw + (4 + ww if words else 0)
            if cx > x and cx + w > x + mid_w:
                cx, cy = x, cy + h + 1
            ir = QRect(cx, cy + (h - ICON) // 2, ICON, ICON)
            nr = QRect(ir.right() + 4, cy, nw + 1, h)
            wr = QRect(nr.right() + 4, cy, ww + 1, h) if words else None
            faint = dim_all or n <= 0
            g["counts"].append((ir, nr, wr, str(icon), str(colour), num,
                                str(word) if words else "", faint))
            tip = f"{n} {word}" if n > 0 and not dim_all else str(word)
            g["count_items"].append((QRect(cx, cy, w, h), tip))
            cx += w + (10 if words else 8)
        for text, kind in rec.get("chips_inline") or ():
            tw = cfm.horizontalAdvance(str(text)) + 14
            if cx > x and cx + tw > x + mid_w:
                cx, cy = x, cy + h + 1
            g["count_chips"].append((QRect(cx, cy + 1, tw, h - 1),
                                     str(text), str(kind or "")))
            cx += tw + 6
        g["lines"]["counts"] = QRect(x, y, mid_w, cy + h - y)
        return cy + h + GAP

    @staticmethod
    def _layout_chips(out: list, chips, x: int, y: int, mid_w: int,
                      chip_fm: QFontMetrics, *, elide: bool = False,
                      tail: str = "") -> int:
        """Lay *chips* (``(text, warn)``) out one under the other from *y*
        into *out*; the y under the last one.

        ``elide`` (Step 3, B): each chip is ONE line, elided in
        the TAIL to the room it has (the head stays readable); the amber
        " -- not the active model" tail is shown only when the whole chip
        fits; the full text is
        the tooltip's (``tip_text``).  ``out`` then holds the shown text."""
        for text, warn in chips:
            inner = max(mid_w - 14, 20)
            if elide:
                text = _elide_tail(chip_fm, text, inner - 2, tail)
                th = chip_fm.height()
            else:
                th = _wrap_height(chip_fm, text, inner)
            # +2: drawText wraps a text drawn at exactly its own
            # advance (fractional metrics), so the chip got a 2nd line
            # its row height never counted (B1b shots)
            tw = min(chip_fm.horizontalAdvance(text) + 2, inner)
            r = QRect(x, y + 1, tw + 14, th + 2)
            out.append((r, text, warn))
            y += r.height() + 2
        return y

    def _row_width(self, option) -> int:
        view = option.widget
        if view is not None and hasattr(view, "viewport"):
            return max(view.viewport().width(), 60)
        return max(option.rect.width(), 60)

    def sizeHint(self, option, index):
        rec = index.data(ROLE_RECORD) or {}
        w = self._row_width(option)
        g = self.layout(QRect(0, 0, w, 0), rec, option.font)
        return QSize(w, g["height"])

    # -- painting ---------------------------------------------------------
    def paint(self, painter: QPainter, option, index):
        rec = index.data(ROLE_RECORD) or {}
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = option.rect
        if not option.state & QStyle.State_Enabled:
            painter.setOpacity(0.55)           # a locked list dims (D3)
        elif self._dim_when_locked and option.widget is not None and                 option.widget.property(self._lock_property):
            painter.setOpacity(0.55)
        if option.state & QStyle.State_Selected:
            painter.fillRect(rect, self._palette["sel_bg"])
        if rec.get("loaded"):
            # A3 fold: the stack / patch LOADED in the viewer -- a faint
            # accent glow over the row and a 2 px accent bar on its left
            # edge; separate from the selection highlight, so it survives
            # a selection change (it rides the record, not the view state)
            accent = QColor(self._palette["accent"])
            glow = QColor(accent)
            glow.setAlphaF(LOADED_GLOW_ALPHA)
            painter.fillRect(rect, glow)
            painter.fillRect(QRect(rect.x(), rect.y(), LOADED_BAR_PX,
                                   rect.height()), accent)
        g = self.layout(rect, rec, option.font)
        f = self._fonts(option.font)
        wide = self._mode == MODE_WIDE
        enabled = bool(rec.get("tickable", True))
        if "tick" in g:
            checked = _is_checked(index.data(Qt.CheckStateRole))
            # B2: a tick the lock took (ItemIsUserCheckable off) dims too
            live = bool(index.flags() & Qt.ItemIsUserCheckable)
            self._paint_tick(painter, g["tick"], checked, enabled and live,
                             accent=self._palette["accent"],
                             mark=self._palette["chk_mark"])
        if "thumb" in g:
            self._paint_thumb(painter, g["thumb"], index.data(ROLE_THUMB),
                              keep_aspect=self._keep_aspect)
        if "icon" in g:
            pm = _icon_pixmap(rec.get("icon") or "mdi6.circle-outline",
                              rec.get("icon_colour") or rec.get("colour")
                              or "#777777")
            if pm is not None:
                painter.drawPixmap(g["icon"].topLeft(), pm)
        view = option.widget
        locked = bool(view is not None
                      and view.property(self._lock_property))
        dim_name = rec.get("dim_name")
        if dim_name is None:
            name_col = _NAME if ((enabled or self._mode == MODE_MEASURE)
                                 and not locked) else _NAME_OFF
        else:
            name_col = _NAME_OFF if (dim_name or locked) else _NAME
        word = rec.get("word") or ""
        dots = rec.get("dots") or ""
        status_text = f"{dots} {word}".strip()
        colour = QColor(rec.get("colour") or "#777777")
        l1 = g["lines"]["l1"]
        if wide:
            painter.setFont(f["word"])
            wfm = QFontMetrics(f["word"])
            ww = wfm.horizontalAdvance(status_text)
            painter.setPen(colour)
            painter.drawText(QRect(l1.right() - ww, l1.y(), ww + 1,
                                   l1.height()),
                             int(Qt.AlignRight | Qt.AlignVCenter), status_text)
            painter.setFont(f["name_wide"])
            painter.setPen(name_col)
            nfm = QFontMetrics(f["name_wide"])
            avail = max(l1.width() - ww - 8, 10)
            painter.drawText(QRect(l1.x(), l1.y(), avail, l1.height()),
                             int(Qt.AlignLeft | Qt.AlignVCenter),
                             nfm.elidedText(rec.get("name") or "",
                                            Qt.ElideMiddle, avail))
            l2 = g["lines"]["l2"]
            painter.setFont(f["dim"])
            dfm = QFontMetrics(f["dim"])
            if rec.get("line2") is not None:
                head = str(rec["line2"])        # Step 3: the known parts
            else:
                head = self._wide_line2.format(
                    axes=rec.get("axes") or "?",
                    c=rec.get("n_channels")
                    if rec.get("n_channels") is not None else "?",
                    spacing="")
            # a record that carries its own line 2 carries its voxel size on
            # line 3; the template's spacing tail is Step 4's old form
            sp = ("" if rec.get("line2") is not None
                  else rec.get("spacing_text") or "")
            painter.setPen(_DIM)
            head_shown = dfm.elidedText(head + sp, Qt.ElideRight, l2.width())
            if rec.get("in_voxels") and head_shown == head + sp:
                hw = dfm.horizontalAdvance(head)
                painter.drawText(l2, int(Qt.AlignLeft | Qt.AlignVCenter), head)
                painter.setPen(_AMBER)
                painter.drawText(QRect(l2.x() + hw, l2.y(),
                                       l2.width() - hw, l2.height()),
                                 int(Qt.AlignLeft | Qt.AlignVCenter), sp)
            else:
                painter.drawText(l2, int(Qt.AlignLeft | Qt.AlignVCenter),
                                 head_shown)
            l3 = g["lines"].get("l3w")
            if l3 is not None:
                # Step 3 (R7): mono, amber when the record says so
                painter.setFont(f["word_small"])
                painter.setPen(self._palette["warn"] if rec.get("line3_warn")
                               else _DIM)
                l3_text = l3 if "action" not in g else QRect(
                    l3.x(), l3.y(), g["action"].left() - l3.x(), l3.height())
                painter.drawText(l3_text, int(Qt.AlignLeft | Qt.AlignVCenter),
                                 QFontMetrics(f["word_small"]).elidedText(
                                     str(rec["line3"]), Qt.ElideRight,
                                     l3_text.width()))
                tail3 = rec.get("line3_tail")
                own_row = g["lines"].get("l3tail")
                if tail3 and own_row is not None:
                    # Step 3 (S1): too long for line 3 -- its own line
                    painter.setPen(self._palette["warn"])
                    painter.drawText(
                        own_row, int(Qt.AlignLeft | Qt.AlignVCenter),
                        QFontMetrics(f["word_small"]).elidedText(
                            str(tail3).lstrip(" ·"), Qt.ElideRight,
                            own_row.width()))
                elif tail3:
                    # Step 3 (S1): the measured-at tail, warn colour
                    sfm3 = QFontMetrics(f["word_small"])
                    used = sfm3.horizontalAdvance(str(rec["line3"]))
                    room = max(l3_text.width() - used, 0)
                    painter.setPen(self._palette["warn"])
                    painter.drawText(
                        QRect(l3_text.x() + used, l3_text.y(), room,
                              l3_text.height()),
                        int(Qt.AlignLeft | Qt.AlignVCenter),
                        sfm3.elidedText(str(tail3), Qt.ElideRight, room))
            if "action" in g:
                # Step 3 (A2): the inline button; dimmed under the lock
                if locked:
                    painter.setOpacity(0.35)
                ar = QRectF(g["action"]).adjusted(0.5, 1.5, -0.5, -1.5)
                painter.setPen(QPen(_CHIP_BORDER, 1))
                painter.setBrush(QColor(255, 255, 255, 18))
                painter.drawRoundedRect(ar, 4, 4)
                painter.setFont(f["chip"])
                painter.setPen(_NAME)
                painter.drawText(g["action"], int(Qt.AlignCenter),
                                 QFontMetrics(f["chip"]).elidedText(
                                     str(rec["row_action"][1]),
                                     Qt.ElideRight,
                                     g["action"].width() - 6))
                if locked:
                    painter.setOpacity(1.0)
            if "eye" in g:
                eye_off = bool(view is not None
                               and view.property(self._eye_lock_property))
                if eye_off:
                    painter.setOpacity(0.35)  # K3: refused under the lock
                self._paint_icon(painter, g["eye"], "mdi6.eye-outline")
                if eye_off:
                    painter.setOpacity(1.0)
            if "menu" in g:
                # R1-8: the menu refuses under the lock, so its glyph dims
                if locked:
                    painter.setOpacity(0.35)
                self._paint_icon(painter, g["menu"], "mdi6.dots-horizontal")
                if locked:
                    painter.setOpacity(1.0)
        else:
            painter.setFont(f["name_mono"])
            painter.setPen(name_col)
            nfm = QFontMetrics(f["name_mono"])
            if rec.get("name_keep_tail"):
                # Step 2 (fix 2): the tail tells the rows apart -- keep it
                name_shown = _elide_keep_tail(
                    nfm, str(rec.get("name") or ""), l1.width(),
                    int(rec["name_keep_tail"]))
            else:
                name_shown = nfm.elidedText(rec.get("name") or "",
                                            Qt.ElideMiddle, l1.width())
            painter.drawText(l1, int(Qt.AlignLeft | Qt.AlignVCenter),
                             name_shown)
            l2 = g["lines"]["l2"]
            if rec.get("line2_plain") is not None:
                # Step 2 (L): one dim text, elided at the tail; (fix 2) a
                # record may offer shorter spellings of the same line
                painter.setFont(f["dim"])
                painter.setPen(_DIM)
                dfm2 = QFontMetrics(f["dim"])
                variants = rec.get("line2_variants")
                if variants:
                    shown2 = _fit_variant(dfm2, variants, l2.width())
                else:
                    shown2 = dfm2.elidedText(str(rec["line2_plain"]),
                                             Qt.ElideRight, l2.width())
                painter.drawText(l2, int(Qt.AlignLeft | Qt.AlignVCenter),
                                 shown2)
            else:
                painter.setFont(f["word_small"])
                painter.setPen(colour)
                left_w = g["axis"].left() - l2.left() - 6
                wfm = QFontMetrics(f["word_small"])
                painter.drawText(QRect(l2.x(), l2.y(), max(left_w, 10),
                                       l2.height()),
                                 int(Qt.AlignLeft | Qt.AlignVCenter),
                                 wfm.elidedText(status_text, Qt.ElideRight,
                                                max(left_w, 10)))
                painter.setPen(_DIM)
                painter.drawText(g["axis"],
                                 int(Qt.AlignLeft | Qt.AlignVCenter),
                                 rec.get("axes") or "")
            if "eye" in g:
                eye_off = bool(view is not None
                               and view.property(self._eye_lock_property))
                if eye_off:
                    painter.setOpacity(0.35)  # K3: refused under the lock
                self._paint_icon(painter, g["eye"], "mdi6.eye-outline")
                if eye_off:
                    painter.setOpacity(1.0)
            if "menu" in g:
                if locked:
                    painter.setOpacity(0.35)
                self._paint_icon(painter, g["menu"], "mdi6.dots-horizontal")
                if locked:
                    painter.setOpacity(1.0)
            if "info" in g:
                # live under the lock: the details can be read (R6)
                self._paint_icon(painter, g["info"],
                                 "mdi6.information-outline")
            if "l3" in g["lines"]:
                painter.setFont(f["dim"])
                painter.setPen(_DIM)
                l3 = g["lines"]["l3"]
                variants = rec.get("shape_variants")
                if variants:
                    painter.drawText(
                        l3, int(Qt.AlignLeft | Qt.AlignVCenter),
                        _fit_variant(QFontMetrics(f["dim"]), variants,
                                     l3.width()))
                else:
                    painter.drawText(l3, int(Qt.TextWordWrap | Qt.AlignLeft),
                                     rec.get("shape_text") or "")
            l3n = g["lines"].get("l3n")
            if l3n is not None:
                # A3: the voxel-size line, mono; amber "voxels" when asked
                painter.setFont(f["word_small"])
                painter.setPen(self._palette["warn"]
                               if rec.get("vox_warn") else _DIM)
                painter.drawText(l3n, int(Qt.TextWordWrap | Qt.AlignLeft),
                                 g["vox_shown"])
            if g.get("notes"):
                painter.setFont(f["dim"])
                painter.setPen(_DIM)
                for r, text in g["notes"]:
                    painter.drawText(r, int(Qt.TextWordWrap | Qt.AlignLeft),
                                     text)
        if "counts" in g:
            self._paint_counts(painter, g, f)
        if "strip" in g:
            self._paint_strip(painter, g["strip"], rec["strip"])
        for dot, nr, wr, num, word, rgb in g.get("tier_draw", ()):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(*rgb[:3]))
            painter.drawEllipse(QRectF(dot))
            painter.setFont(f["num"])
            painter.setPen(_NAME)
            painter.drawText(nr, int(Qt.AlignLeft | Qt.AlignVCenter), num)
            if word:
                painter.setFont(f["dim"])
                painter.setPen(_DIM)
                painter.drawText(wr, int(Qt.AlignLeft | Qt.AlignVCenter),
                                 word)
        if "tier_ellipsis" in g:
            painter.setFont(f["dim"])
            painter.setPen(_DIM)
            painter.drawText(g["tier_ellipsis"],
                             int(Qt.AlignLeft | Qt.AlignVCenter), "…")
        painter.setFont(f["dim"])
        painter.setPen(self._palette["warn"])
        for r, text in g["warns"]:
            painter.drawText(r, int(Qt.TextWordWrap | Qt.AlignLeft), text)
        painter.setFont(f["chip"])
        for r, text, warn in g.get("chips_top", []) + g["chips"]:
            painter.setPen(QPen(_CHIP_WARN_BORDER if warn else _CHIP_BORDER,
                                1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5),
                                    8, 8)
            painter.setPen(self._palette["warn"] if warn else _CHIP_TEXT)
            painter.drawText(r.adjusted(7, 1, -7, -1),
                             int(Qt.TextWordWrap | Qt.AlignLeft
                                 | Qt.AlignVCenter), text)
        painter.restore()

    @staticmethod
    def _paint_counts(painter, g: dict, f: dict):
        """Step 2's icon-counts line (L): the icon in the status colour, the
        number (blank at zero), the word on a wide row, the inline chips; a
        faint slot is drawn at 35 %; ``grey_word`` is the one grey word."""
        if "grey_word" in g:
            r, text = g["grey_word"]
            painter.setFont(f["dim"])
            painter.setPen(QColor("#777777"))
            painter.drawText(r, int(Qt.AlignLeft | Qt.AlignVCenter), text)
        for ir, nr, wr, icon, colour, num, word, faint in g["counts"]:
            if faint:
                painter.setOpacity(painter.opacity() * 0.35)
            pm = _icon_pixmap(icon, colour)
            if pm is not None:
                painter.drawPixmap(ir.topLeft(), pm)
            if num:
                painter.setFont(f["num"])
                painter.setPen(_NAME)
                painter.drawText(nr, int(Qt.AlignLeft | Qt.AlignVCenter), num)
            if wr is not None and word:
                painter.setFont(f["dim"])
                painter.setPen(QColor(colour) if not faint else _DIM)
                painter.drawText(wr, int(Qt.AlignLeft | Qt.AlignVCenter),
                                 word)
            if faint:
                painter.setOpacity(painter.opacity() / 0.35)
        painter.setFont(f["chip"])
        for r, text, kind in g["count_chips"]:
            round_chip = kind == "round"
            edge = QColor("#8c9eff") if round_chip else QColor(_CHIP_BORDER)
            painter.setPen(QPen(edge, 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(r).adjusted(0.5, 0.5, -0.5, -0.5),
                                    8, 8)
            painter.setPen(QColor("#8c9eff") if round_chip else (
                QColor("#666666") if kind == "dim" else _CHIP_TEXT))
            painter.drawText(r, int(Qt.AlignCenter), text)

    @staticmethod
    def _paint_strip(painter, r: QRect, segments):
        """Step 2's proportion strip (L): ``(fraction, colour)`` segments
        left to right across *r*; the last one takes the rounding."""
        painter.save()
        painter.setPen(Qt.NoPen)
        x = r.x()
        total = r.width()
        segs = [(max(float(fr), 0.0), col) for fr, col in segments]
        used = 0.0
        for i, (fr, col) in enumerate(segs):
            used += fr
            end = r.x() + total if i == len(segs) - 1 else                 r.x() + int(round(min(used, 1.0) * total))
            if end > x:
                painter.setBrush(QColor(col))
                painter.drawRect(QRect(x, r.y(), end - x, r.height()))
                x = end
        painter.restore()

    @staticmethod
    def _paint_tick(painter, r: QRect, checked: bool, enabled: bool, *,
                    accent=None, mark=None):
        """The one tick of both steps (A3, rows mockup v3 ``.tick``): a 2 px
        grey border and no fill unticked; ticked, the box is filled with the
        list's *accent* (default ``STEP4_ACCENT``) with a ``TICK_HALO`` px
        halo of the accent at 35 % outside it and the tick glyph in *mark*
        (default white).  A tick the row cannot take dims."""
        accent = QColor(accent if accent is not None else STEP4_ACCENT)
        mark = QColor(mark if mark is not None else "#ffffff")
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        if not enabled:
            painter.setOpacity(0.45)
        if checked:
            halo = QColor(accent)
            halo.setAlphaF(TICK_HALO_ALPHA)
            painter.setPen(QPen(halo, TICK_HALO))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(
                QRectF(r).adjusted(-TICK_HALO / 2.0, -TICK_HALO / 2.0,
                                   TICK_HALO / 2.0, TICK_HALO / 2.0),
                3 + TICK_HALO / 2.0, 3 + TICK_HALO / 2.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(accent)
            painter.drawRoundedRect(QRectF(r), 3, 3)
            pen = QPen(mark, 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            path = QPainterPath()
            path.moveTo(r.x() + 4, r.y() + 8.5)
            path.lineTo(r.x() + 7, r.y() + 11.5)
            path.lineTo(r.x() + 12.5, r.y() + 5)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        else:
            painter.setPen(QPen(_CHK_BORDER, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(r).adjusted(1, 1, -1, -1), 3, 3)
        painter.restore()

    @staticmethod
    def _paint_thumb(painter, r: QRect, pixmap, keep_aspect: bool = False):
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(_THUMB_BG)
        painter.drawRoundedRect(QRectF(r), 3, 3)
        if pixmap is not None and not pixmap.isNull():
            target = r
            pw, ph = pixmap.width(), pixmap.height()
            if keep_aspect and pw > 0 and ph > 0 and pw != ph:
                # fitted inside the square, centred (the dark square shows
                # around a slice that is not square)
                k = min(r.width() / pw, r.height() / ph)
                w, h = max(1, round(pw * k)), max(1, round(ph * k))
                target = QRect(r.x() + (r.width() - w) // 2,
                               r.y() + (r.height() - h) // 2, w, h)
            painter.drawPixmap(target, pixmap)
        else:
            painter.setBrush(_THUMB_IN)
            painter.drawRoundedRect(QRectF(r.adjusted(3, 3, -3, -3)), 2, 2)
        painter.restore()

    @staticmethod
    def _paint_icon(painter, r: QRect, name: str):
        pm = _icon_pixmap(name, _ICON_DIM)
        if pm is not None:
            painter.drawPixmap(r.x() + 1, r.y() + 1, pm)

    # -- clicks and help --------------------------------------------------
    def _geometry_at(self, option, index):
        rec = index.data(ROLE_RECORD) or {}
        return self.layout(option.rect, rec, option.font), rec

    def editorEvent(self, event, model, option, index):
        et = event.type()
        if et not in (QEvent.MouseButtonRelease, QEvent.MouseButtonPress,
                      QEvent.MouseButtonDblClick):
            return super().editorEvent(event, model, option, index)
        if event.button() != Qt.LeftButton:
            return super().editorEvent(event, model, option, index)
        g, rec = self._geometry_at(option, index)
        pos = event.pos()
        name = index.data(ROLE_NAME) or rec.get("name") or ""
        tick = g.get("tick")
        if tick is not None and tick.adjusted(-3, -3, 3, 3).contains(pos):
            flags = index.flags()
            if not (flags & Qt.ItemIsUserCheckable) or not (
                    flags & Qt.ItemIsEnabled):
                return True
            if et == QEvent.MouseButtonRelease:
                cur = _is_checked(index.data(Qt.CheckStateRole))
                if not cur and not rec.get("tickable", True):
                    # a greyed row can be unticked, never
                    # ticked
                    return True
                model.setData(index,
                              Qt.Unchecked if cur else Qt.Checked,
                              Qt.CheckStateRole)
            return True
        eye = g.get("eye")
        if eye is not None and eye.adjusted(-2, -2, 2, 2).contains(pos):
            if et == QEvent.MouseButtonRelease:
                print(f"[STEP4-UI] eye {name}")
                self.eye_clicked.emit(name)
            return True
        info = g.get("info")
        if info is not None and info.adjusted(-2, -2, 2, 2).contains(pos):
            if et == QEvent.MouseButtonRelease:
                print(f"[STEP3-UI] info {name}")
                self.info_clicked.emit(name)
            return True
        act = g.get("action")
        if act is not None and act.contains(pos):
            if et == QEvent.MouseButtonRelease:
                key = str((rec.get("row_action") or ("",))[0])
                print(f"[STEP3-UI] action {key} {name}")
                self.action_clicked.emit(name, key)
            return True
        menu = g.get("menu")
        if menu is not None and menu.adjusted(-2, -2, 2, 2).contains(pos):
            if et == QEvent.MouseButtonRelease:
                view = option.widget
                gpos = (view.viewport().mapToGlobal(menu.bottomLeft())
                        if view is not None else event.globalPos())
                print(f"[STEP4-UI] menu {name}")
                self.menu_requested.emit(name, gpos)
            return True
        return super().editorEvent(event, model, option, index)

    def help_key_at(self, option, index, pos) -> Optional[str]:
        """Which ``help_lines`` key the painted part under *pos* has."""
        if self._mode == MODE_MEASURE:
            g, _rec = self._geometry_at(option, index)
            tl = g["lines"].get("tiers")
            if tl is not None and tl.contains(pos):
                return "row_tiers"
            return "mt_row_click"
        g, _rec = self._geometry_at(option, index)
        vox = g["lines"].get("l3n")
        if vox is not None and vox.contains(pos):
            return "row_vox"
        tick = g.get("tick")
        if tick is not None and tick.adjusted(-3, -3, 3, 3).contains(pos):
            return "row_tick"
        eye = g.get("eye")
        if eye is not None and eye.adjusted(-2, -2, 2, 2).contains(pos):
            return "row_eye"
        info = g.get("info")
        if info is not None and info.adjusted(-2, -2, 2, 2).contains(pos):
            return "row_info"
        act = g.get("action")
        if act is not None and act.contains(pos):
            return "row_action"
        tl = g["lines"].get("tiers")
        if tl is not None and tl.contains(pos):
            return "row_tiers"
        cl = g["lines"].get("counts")
        if cl is not None and cl.contains(pos):
            return "row_counts"
        for r, _t, _w in g.get("chips_top", ()):
            if r.contains(pos):
                return "row_made_by"
        thumb = g.get("thumb")
        if (thumb is not None and thumb.contains(pos)
                and index.data(ROLE_THUMB_TIP)):
            return "row_thumb"            # Step 3: the picture's own tip
        for r, _t, _w in g["chips"]:
            if r.contains(pos):
                return "row_chip"
        return None

    def tip_text(self, option, index, pos):
        """The tooltip of the painted part under *pos* (None: none)."""
        key = self.help_key_at(option, index, pos)
        rec = index.data(ROLE_RECORD) or {}
        if key == "row_thumb":
            text = index.data(ROLE_THUMB_TIP)
        elif key == "row_vox":
            # with its source; B: a record's own sentence for a patch with
            # no voxel size ("cell width falls back to pixels")
            text = str(rec.get("vox_tip") or rec.get("vox_text") or "")
        elif key == "row_made_by":
            # B: the chip is elided at the tail on a narrow
            # row -- the tooltip carries the whole name
            text = self._help_lines.get(key)
            top = rec.get("chips_top") or ()
            if top and rec.get("chips_top_elide"):
                full = str(top[0][0])
                text = (full + chr(10) + text) if text else full
        elif key == "row_tiers":
            text = self._help_lines.get(key)
            g, _rec = self._geometry_at(option, index)
            compact = not g.get("tier_draw") or not g["tier_draw"][0][4]
            if compact or not text:
                # A3: a compact tier line draws no word -- the tooltip
                # names the tier under the pointer (and the line's help);
                # a line with words but no help line of its own names it too
                for box, word in g.get("tier_items", ()):
                    if box.contains(pos):
                        text = (word + chr(10) + text) if (
                            text and compact) else (text or word)
                        break
        elif key == "row_counts":
            # Step 2 (L): the slot under the pointer says its word and count
            text = self._help_lines.get(key)
            g, _rec = self._geometry_at(option, index)
            for box, tip in g.get("count_items", ()):
                if box.contains(pos):
                    text = tip
                    break
        elif key == "row_chip" and rec.get("chips_elide"):
            # an elided chip: the row's own sentence carries the whole text
            text = rec.get("row_tip") or self._help_lines.get(key)
        else:
            text = self._help_lines.get(key) if key is not None else None
        if not text and key is None:
            # a row's own sentence (Step 3: why a row cannot be ticked)
            text = rec.get("row_tip")
        return text

    def helpEvent(self, event, view, option, index):
        if event is None or event.type() != QEvent.ToolTip:
            return super().helpEvent(event, view, option, index)
        text = self.tip_text(option, index, event.pos())
        if not text:
            QToolTip.hideText()
            return True
        QToolTip.showText(event.globalPos(), text, view)
        return True


class StackListWidget(QListWidget):
    """A ``QListWidget`` (D-a: the controller and the shell read list-widget
    methods) whose rows re-lay out on a resize, because the delegate's row
    height depends on the width; it paints its empty-state lines itself."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._empty_lines: List[str] = []
        #: opt-in (Step 2): see :meth:`paintEvent`
        self.empty_one_line_when_short = False
        self.setUniformItemSizes(False)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWordWrap(True)

    def set_empty_lines(self, lines) -> None:
        self._empty_lines = [str(x) for x in lines or ()]
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.doItemsLayout()
        # the viewport can settle after this event (a section opening,
        # the theme's font arriving): lay the rows out again once it has
        QTimer.singleShot(0, self._relayout)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._relayout)

    def _relayout(self):
        try:
            self.doItemsLayout()
        except RuntimeError:          # deleted before the timer fired
            pass

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() == 0 and self._empty_lines:
            p = QPainter(self.viewport())
            f = QFont(self.font())
            f.setPixelSize(11)
            p.setFont(f)
            p.setPen(_DIM)
            r = self.viewport().rect().adjusted(16, 22, -16, -8)
            flags = int(Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap)
            text = "\n".join(self._empty_lines)
            if self.empty_one_line_when_short:
                # Step 2 (fix 2): a viewport too short for the whole message
                # shows its first line alone, near the top, and the whole
                # message on hover -- never a clipped row of dots
                need = QFontMetrics(f).boundingRect(
                    QRect(0, 0, max(r.width(), 10), 10000), flags, text
                ).height()
                if need > r.height():
                    r = self.viewport().rect().adjusted(8, 6, -8, -4)
                    # (polish) the whole message when it fits the tighter
                    # rectangle -- the words return whenever the column is
                    # wide enough for them
                    tight = QFontMetrics(f).boundingRect(
                        QRect(0, 0, max(r.width(), 10), 10000), flags, text
                    ).height()
                    if tight <= r.height():
                        p.drawText(r, flags, text)
                        p.end()
                        self.viewport().setToolTip("")
                        return
                    first = QFontMetrics(f).elidedText(
                        self._empty_lines[0], Qt.ElideRight, r.width())
                    p.drawText(r, int(Qt.AlignHCenter | Qt.AlignTop), first)
                    p.end()
                    self.viewport().setToolTip(text)
                    return
                self.viewport().setToolTip("")
            p.drawText(r, flags, text)
            p.end()
