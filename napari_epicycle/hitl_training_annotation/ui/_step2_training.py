"""Step 2 look pass, brick T (module half): the Training section's builders.

Everything here
RECEIVES the widgets ``main_widget.py`` constructs as literal ``self.x =``
assignments (the lock list reads that source text) and creates only the NEW
ones: ``train_cancel_btn`` is passed in by the caller too (it is a stored
attribute the controller reads), the foot splitter, the Loss collapsible, the
lock line, the stage-line label, the help rings.

Public surface:

* ``build_setup_column`` -> ``SetupColumn`` (Writes, Mode + hint, Starting
  model + file, Epochs / Batch size, Device, Advanced);
* ``build_staged_column`` -> ``StagedColumn``;
* ``build_training_foot`` -> ``TrainingFoot`` (Train . Cancel, bar, stage line,
  Loss collapsible);
* ``build_training_panel`` -> ``TrainingPanel`` (lock line, the two columns in
  scroll areas, the foot on a vertical splitter);
* ``LockLine``, ``LossCollapsible``;
* pure: ``compose_training_stage_line``, ``hint_text``, ``writes_text``,
  ``staged_count_text``, ``staged_other_text``.

The sentences and the pure text functions live in ``_step2_text.py`` and are
re-exported here under their names.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from napari_epicycle.hitl_training_annotation.ui._step2_curation import (  # noqa: E501
    CompactSplitter, ElidedLabel, SNAP_SLACK, SnapWatcher,
    enclosing_viewport_height, snap_cap,
)
from napari_epicycle.ui_shell.theme import (
    MONO_LABEL_STYLE,
    PANEL_HELP_STYLE,
    PANEL_GOLD,
    PANEL_LEFT_INNER_MARGIN,
    PANEL_PROGRESS_H,
    PANEL_PROGRESS_H_SHORT,
    PANEL_RIGHT_INNER_MARGIN,
    PANEL_ROW_LABEL_STYLE,
    PANEL_SPLIT_HANDLE,
    PANEL_WARN_STYLE,
    PRIMARY_BTN_STYLE,
    SECONDARY_BTN_STYLE,
    SPLITTER_STYLE,
    apply_button_icon,
    disclosure_btn_style,
    panel_foot_style,
    progress_bar_style,
)
from napari_epicycle.ui_shell._recon_settings_widgets import (
    ENGINE_NAME_STYLE,
    LockLine as _SharedLockLine,
)

from napari_epicycle.hitl_training_annotation.ui._step2_text import (  # noqa: F401  (re-exported)
    TRAIN_STAGED_TITLE,
    TRAIN_SETUP_TITLE,
    STAGED_DESC,
    STAGED_COUNT,
    STAGED_COUNT_FIRST,
    STAGED_OTHER_NONE,
    STAGED_OTHER_SOME,
    SETUP_DESC,
    WRITES_LABEL,
    WRITES_LINE,
    WRITES_OTHER_IDLE,
    MODE_LABEL,
    MODE_CONTINUE,
    MODE_SCRATCH,
    STARTING_MODEL_LABEL,
    STARTING_FILE_LABEL,
    STARTING_NOTE,
    STARTING_GREY_REASON,
    EPOCHS_LABEL,
    BATCH_LABEL,
    SETUP_ENGINE_NAMES,
    DEVICE_LABEL,
    ADVANCED_TITLE,
    TRAIN_LOCK_LINE,
    TRAIN_BTN_TEXT,
    CANCEL_BTN_TEXT,
    STAGE_BAR_FORMAT,
    STAGE_IDLE,
    STAGE_IDLE_BARE,
    STAGE_RUNNING,
    STAGE_STARTING,
    LOSS_TITLE,
    LOSS_LIVE_CAPTION,
    HINT_ROUND0_SCRATCH,
    HINT_ROUND0_LOCKED,
    HINT_CUMULATIVE_PRIOR,
    HINT_CUMULATIVE_MISSING,
    HINT_SCRATCH_FILE,
    HINT_SCRATCH_BASE,
    HINT_CASES,
    hint_text,
    writes_text,
    _cap,
    staged_count_text,
    staged_other_text,
    compose_training_stage_line,
    round_from_name,
)

# -- numbers (the Model Refinement ones) -----------------------------

LEFT_MIN_W = 240
RIGHT_MIN_W = 300
SPLIT_TOP_MIN_H = 100          # the columns can shrink this far (218 px room)
LOSS_MIN_H = 140               # the open plot's floor
SPLIT_OPEN_FRAC = 0.55         # the foot's share of the section when Loss opens
SPLIT_HANDLE = PANEL_SPLIT_HANDLE
PROGRESS_H = PANEL_PROGRESS_H
FOOT_OBJECT = "step2TrainingFoot"
LOSS_TOGGLE_OBJECT = "step2LossToggle"
#: the Loss header's arrows (superqt's own pair, as Step 3's collapsibles)
LOSS_ARROW_CLOSED = "\u25b6"
LOSS_ARROW_OPEN = "\u25bc"


# -- small widgets -------------------------------------------------------------


def _label(text: str, parent: QWidget, style: str = PANEL_ROW_LABEL_STYLE,
           wrap: bool = False) -> QLabel:
    lab = QLabel(text, parent)
    lab.setStyleSheet(style)
    lab.setWordWrap(wrap)
    if wrap:
        lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return lab


def _label_with_engine(text: str, engine: str, parent: QWidget) -> QWidget:
    """``Epochs  n_epochs``: the plain label with the engine's own name beside
    it, dim and mono (the mockup's tag; the Advanced rows do the same)."""
    box = QWidget(parent)
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    box.plain_label = _label(text, box)
    box.engine_label = QLabel(engine, box)
    box.engine_label.setStyleSheet(ENGINE_NAME_STYLE)
    h.addWidget(box.plain_label)
    h.addWidget(box.engine_label)
    h.addStretch(1)
    return box


def _header(parent: QWidget, title: str, *, extra: Optional[QWidget] = None,
            help_factory: Optional[Callable[[QWidget], QWidget]] = None) -> QHBoxLayout:
    from napari_epicycle.ui_shell.theme import SECTION_HEADER_STYLE
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    lab = QLabel(title, parent)
    lab.setStyleSheet(SECTION_HEADER_STYLE)
    row.addWidget(lab)
    row.title_label = lab
    row.addStretch(1)
    if extra is not None:
        row.addWidget(extra)
    if help_factory is not None:
        row.addWidget(help_factory(parent))
    return row


class LockLine(_SharedLockLine):
    """The S5 lock sentence; hidden while idle.  The panel's shared lock line
    (left edge, lock icon, 11 px -- Session's and Curation's), plus
    ``set(locked, round_name, bucket)``, the one call ``on_training_locked``
    makes."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, object_name="step2TrainLockLine")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def set(self, locked: bool, round_name: str = "", bucket: str = "") -> None:
        if not locked:
            self.setText("")
            return
        n = round_from_name(round_name)
        self.setText(TRAIN_LOCK_LINE.format(n="?" if n is None else n,
                                            bucket=bucket or "lateral"))


class LossCollapsible(QWidget):
    """Header (arrow, title, caption, optional view combo) over a body that
    holds the loss plot.  Own class, not superqt's, so the open body is never
    capped (superqt's ``QCollapsible`` caps its content).  Closed idle the
    caption is ``loss_plot.grading_caption()``."""

    toggled = Signal(bool)

    def __init__(self, loss_plot: QWidget, view_combo: Optional[QWidget] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._plot = loss_plot
        self._live_caption = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QWidget(self)
        h = QHBoxLayout(head)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        # the header is Step 3's boxed disclosure bar (COLLAPSIBLE_STYLE's
        # values), arrow + title, the caption inside it on the right
        self.toggle_btn = QPushButton(f"{LOSS_ARROW_CLOSED}  {LOSS_TITLE}",
                                      head)
        self.toggle_btn.setObjectName(LOSS_TOGGLE_OBJECT)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet(disclosure_btn_style(LOSS_TOGGLE_OBJECT))
        self.toggle_btn.setSizePolicy(QSizePolicy.Expanding,
                                      QSizePolicy.Fixed)
        self.toggle_btn.toggled.connect(self._on_toggle)
        self.caption_label = QLabel("", self.toggle_btn)
        self.caption_label.setStyleSheet(ENGINE_NAME_STYLE)
        self.caption_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        cap_row = QHBoxLayout(self.toggle_btn)
        cap_row.setSizeConstraint(QHBoxLayout.SetNoConstraint)
        cap_row.setContentsMargins(0, 0, 8, 0)
        cap_row.addStretch(1)
        cap_row.addWidget(self.caption_label, 0,
                          Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self.toggle_btn, 1)
        self.view_combo = view_combo
        if view_combo is not None:
            h.addWidget(view_combo)
        lay.addWidget(head)
        self.body = QWidget(self)
        b = QVBoxLayout(self.body)
        b.setContentsMargins(0, 2, 0, 0)
        b.setSpacing(0)
        loss_plot.setParent(self.body)
        loss_plot.setMinimumHeight(LOSS_MIN_H)
        loss_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        b.addWidget(loss_plot, 1)
        lay.addWidget(self.body, 1)
        self.body.setVisible(False)
        self._apply_policy(False)
        self.refresh_caption()

    # state
    def is_open(self) -> bool:
        return self.toggle_btn.isChecked()

    def _apply_policy(self, opened: bool) -> None:
        self.setSizePolicy(QSizePolicy.Preferred,
                           QSizePolicy.Expanding if opened else QSizePolicy.Maximum)

    def _on_toggle(self, opened: bool) -> None:
        self.toggle_btn.setText(
            f"{LOSS_ARROW_OPEN if opened else LOSS_ARROW_CLOSED}  {LOSS_TITLE}")
        self.body.setVisible(opened)
        self._apply_policy(opened)
        self.toggled.emit(opened)

    # caption
    def refresh_caption(self) -> None:
        """Idle caption from the plot (the last round); a live one wins."""
        if self._live_caption:
            self.caption_label.setText(self._live_caption)
            return
        fn = getattr(self._plot, "grading_caption", None)
        try:
            self.caption_label.setText(fn() if callable(fn) else "")
        except Exception:      # noqa: BLE001 - a caption must not break the UI
            self.caption_label.setText("")

    def set_live_caption(self, text: str) -> None:
        self._live_caption = text or ""
        self.refresh_caption()

    # hooks
    def open_for_round(self, caption: str = "") -> None:
        """A round started: show the plot and the live caption."""
        if caption:
            self._live_caption = caption
        self.refresh_caption()
        self.toggle_btn.setChecked(True)

    def close_idle(self) -> None:
        """The round ended: back to closed, last round in the header."""
        self._live_caption = ""
        self.toggle_btn.setChecked(False)
        self.refresh_caption()


_QWIDGETSIZE_MAX = 16777215   # Qt's "no maximum"
#: the shortest viewport a snap may leave (a header and a line)
SNAP_MIN_H = 60


class _Scroll(QScrollArea):
    def __init__(self, content: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        content.setParent(self)
        self.setWidget(content)

    def snap_viewport(self, room: int) -> None:
        """End the viewport at a block boundary (a header, a row, a hint --
        the content layout's own items) so the first screenful never shows a
        half-cut line above the pinned foot; the scroll bar still reaches the
        rest.  ``room <= 0`` or content that fits: no cap."""
        content = self.widget()
        lay = content.layout() if content is not None else None
        if lay is None or room <= 0:
            self.setMaximumHeight(_QWIDGETSIZE_MAX)
            return
        lay.activate()
        bottoms = []
        for i in range(lay.count()):
            item = lay.itemAt(i)
            if item.spacerItem() is not None:
                continue
            w = item.widget()
            if w is not None and w.isHidden():
                continue
            g = item.geometry()
            if g.height() <= 0:
                continue
            bottoms.append(g.bottom() + 1)
        bottoms.sort()
        if not bottoms or bottoms[-1] + lay.contentsMargins().bottom() <= room:
            self.setMaximumHeight(_QWIDGETSIZE_MAX)         # it all fits
            return
        chrome = snap_cap(self, 0) - SNAP_SLACK
        fits = [b for b in bottoms
                if SNAP_MIN_H <= b and b + chrome <= room]
        self.setMaximumHeight(
            snap_cap(self, max(fits)) if fits else _QWIDGETSIZE_MAX)


# -- columns -------------------------------------------------------------------


class SetupColumn(QWidget):
    """The right column: Writes, Mode + hint, Starting model + file, Epochs /
    Batch size, Device, Advanced.  Holds references to the received widgets."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._override = False
        self._prior_enabled: dict = {}
        self._locked = False
        self._lock_prior: dict = {}
        #: widgets the training lock greys that the controller's lock list
        #: does not own (filled by the caller; each keeps its own prior state)
        self.lock_extra: list = []

    def set_hint(self, severity: str, text: str) -> None:
        """ONE hint; the amber block only for ``warn`` (the sanctioned amber)."""
        self.hint_label.setText(text or "")
        self.hint_label.setStyleSheet(
            PANEL_WARN_STYLE if severity == "warn" else PANEL_HELP_STYLE)
        self.hint_label.setVisible(bool(text))
        self.set_override(severity == "override")

    def set_override(self, override: bool) -> None:
        """Grey Starting model + file iff the hint's first value is "override"
        (D19).  Records each widget's prior enabled state and restores it.
        Under the training lock only the flag and the reason move; the lock
        owns the enabled state and ``set_locked(False)`` re-asserts this."""
        override = bool(override)
        if override == self._override:
            return
        self._override = override
        self.grey_reason_label.setVisible(override)
        if self._locked:
            return
        widgets = self._grey_widgets
        if override:
            self._prior_enabled = {id(w): w.isEnabled() for w in widgets}
            for w in widgets:
                w.setEnabled(False)
        else:
            for w in widgets:
                w.setEnabled(self._prior_enabled.get(id(w), True))
            self._prior_enabled = {}

    def is_override(self) -> bool:
        return self._override

    def set_locked(self, locked: bool) -> None:
        """The training lock (S5): grey ``lock_extra`` per widget (never a
        container, so the ``?`` and Advanced stay live), remembering each
        widget's own prior state; on unlock restore them and re-assert the
        override grey the controller's re-enable may have lifted."""
        locked = bool(locked)
        if locked == self._locked:
            return
        self._locked = locked
        widgets = self._grey_widgets
        if locked:
            self._lock_prior = {id(w): w.isEnabled() for w in self.lock_extra}
            for w in self.lock_extra:
                w.setEnabled(False)
            return
        for w in self.lock_extra:
            w.setEnabled(self._lock_prior.get(id(w), True))
        self._lock_prior = {}
        if self._override:
            for w in widgets:
                w.setEnabled(False)
        elif self._prior_enabled:
            for w in widgets:
                w.setEnabled(self._prior_enabled.get(id(w), True))
            self._prior_enabled = {}

    def is_locked(self) -> bool:
        return self._locked


def build_setup_column(
    *,
    parent: Optional[QWidget] = None,
    model_name_lateral_label: QWidget,
    model_name_axial_label: QWidget,
    training_mode_combo,
    model_selector,
    starting_checkpoint_path,
    starting_checkpoint_browse,
    starting_checkpoint_status: Optional[QWidget] = None,
    epochs_sb,
    batch_sb,
    device_gpu,
    device_cpu,
    advanced_collapsible: Optional[QWidget] = None,
    history_button: Optional[QWidget] = None,
    help_factory: Optional[Callable[[QWidget], QWidget]] = None,
    writes_line_text: str = "",
) -> SetupColumn:
    """Place the received Training-setup widgets.  Mode item texts
    are rewritten in place to the D15 words (the combo keeps its two items in
    order, so the controller's substring tests keep working)."""
    col = SetupColumn(parent)
    lay = QVBoxLayout(col)
    lay.setContentsMargins(PANEL_RIGHT_INNER_MARGIN, 6, 10, 6)
    lay.setSpacing(5)
    _h = _header(col, TRAIN_SETUP_TITLE, extra=history_button,
                 help_factory=help_factory)
    lay.addLayout(_h)
    col.title_label = _h.title_label
    col.desc_label = _label(SETUP_DESC, col, PANEL_HELP_STYLE, wrap=True)
    lay.addWidget(col.desc_label)

    # Writes
    wr = QHBoxLayout()
    wr.setSpacing(6)
    wr.addWidget(_label(WRITES_LABEL, col))
    names = QVBoxLayout()
    names.setSpacing(1)
    for w in (model_name_lateral_label, model_name_axial_label):
        w.setParent(col)
        names.addWidget(w)
    wr.addLayout(names, 1)
    lay.addLayout(wr)

    # Mode
    if training_mode_combo.count() >= 2:
        training_mode_combo.setItemText(0, MODE_CONTINUE)
        training_mode_combo.setItemText(1, MODE_SCRATCH)
    training_mode_combo.setParent(col)
    lay.addWidget(_label(MODE_LABEL, col))
    lay.addWidget(training_mode_combo)
    col.hint_label = _label("", col, PANEL_HELP_STYLE, wrap=True)
    col.hint_label.setVisible(False)
    lay.addWidget(col.hint_label)

    # Starting model + file
    col.starting_model_label = _label_with_engine(
        STARTING_MODEL_LABEL, SETUP_ENGINE_NAMES["starting_model"], col)
    lay.addWidget(col.starting_model_label)
    model_selector.setParent(col)
    lay.addWidget(model_selector)
    lay.addWidget(_label(STARTING_FILE_LABEL, col))
    row = QHBoxLayout()
    row.setSpacing(6)
    starting_checkpoint_path.setParent(col)
    starting_checkpoint_browse.setParent(col)
    row.addWidget(starting_checkpoint_path, 1)
    row.addWidget(starting_checkpoint_browse)
    lay.addLayout(row)
    if starting_checkpoint_status is not None:
        starting_checkpoint_status.setParent(col)
        lay.addWidget(starting_checkpoint_status)
    lay.addWidget(_label(STARTING_NOTE, col, PANEL_HELP_STYLE, wrap=True))
    col.grey_reason_label = _label(STARTING_GREY_REASON, col, PANEL_HELP_STYLE, wrap=True)
    col.grey_reason_label.setVisible(False)
    lay.addWidget(col.grey_reason_label)
    col._grey_widgets = [model_selector, starting_checkpoint_path,
                         starting_checkpoint_browse]

    # Epochs / Batch size side by side, label over field
    eb = QHBoxLayout()
    eb.setSpacing(8)
    col.engine_labels = []
    for text, key, w in ((EPOCHS_LABEL, "epochs", epochs_sb),
                         (BATCH_LABEL, "batch", batch_sb)):
        box = QVBoxLayout()
        box.setSpacing(2)
        tag = _label_with_engine(text, SETUP_ENGINE_NAMES[key], col)
        col.engine_labels.append(tag)
        box.addWidget(tag)
        w.setParent(col)
        box.addWidget(w)
        eb.addLayout(box, 1)
    lay.addLayout(eb)

    # Device as gold radios (the root sheet carries PANEL_RADIO_STYLE)
    dev = QHBoxLayout()
    dev.setSpacing(10)
    dev.addWidget(_label(DEVICE_LABEL, col))
    for r in (device_gpu, device_cpu):
        r.setParent(col)
        dev.addWidget(r)
    dev.addStretch(1)
    lay.addLayout(dev)

    if advanced_collapsible is not None:
        advanced_collapsible.setParent(col)
        lay.addWidget(advanced_collapsible)
        try:
            advanced_collapsible.collapse(animate=False)     # closed
        except Exception:      # noqa: BLE001 - any QWidget is accepted
            pass
    lay.addStretch(1)
    col.mode_combo = training_mode_combo
    col.model_selector = model_selector
    col.checkpoint_path = starting_checkpoint_path
    col.checkpoint_browse = starting_checkpoint_browse
    return col


def _col_set_short(col, short: bool) -> None:
    """Short mode (Curation's rule): the description is the header's tooltip."""
    col._short = bool(short)
    col.desc_label.setVisible(not short)
    col.title_label.setToolTip(col.desc_label.text() if short else "")


class StagedColumn(QWidget):
    """The left column: header, desc, the staged list, the honest count."""

    def set_desc(self, bucket: str) -> None:
        self.desc_label.setText(STAGED_DESC.format(bucket=bucket))
        if getattr(self, "_short", False):
            self.title_label.setToolTip(self.desc_label.text())

    def set_short(self, short: bool) -> None:
        _col_set_short(self, short)


def build_staged_column(
    data_to_add_list: QWidget,
    count_label: QLabel,
    *,
    parent: Optional[QWidget] = None,
    bucket: str = "lateral",
    help_factory: Optional[Callable[[QWidget], QWidget]] = None,
) -> StagedColumn:
    """Place the staged list (rows are painted by brick A) and its count."""
    col = StagedColumn(parent)
    lay = QVBoxLayout(col)
    lay.setContentsMargins(0, 6, PANEL_LEFT_INNER_MARGIN, 6)
    lay.setSpacing(5)
    _h = _header(col, TRAIN_STAGED_TITLE, help_factory=help_factory)
    lay.addLayout(_h)
    col.title_label = _h.title_label
    col.desc_label = _label("", col, PANEL_HELP_STYLE, wrap=True)
    lay.addWidget(col.desc_label)
    col.set_desc(bucket)
    data_to_add_list.setParent(col)
    data_to_add_list.setMinimumHeight(60)
    lay.addWidget(data_to_add_list, 1)
    count_label.setParent(col)
    count_label.setWordWrap(True)
    count_label.setStyleSheet(PANEL_ROW_LABEL_STYLE)
    count_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    lay.addWidget(count_label)
    col.list = data_to_add_list
    col.count_label = count_label
    return col


# -- the foot ------------------------------------------------------------------


class TrainingFoot(QWidget):
    """Train . Cancel, the bar, the stage line, the Loss collapsible."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.on_content_changed = None      # the panel re-fits the foot
        self._short = False

    def set_short(self, short: bool) -> None:
        """A short dock: the stage sentence is ONE elided line (the full text
        is its tooltip) and the bar is 8 px (the columns take the room); the
        tall foot wraps the sentence and keeps the 14 px bar."""
        short = bool(short)
        self._short = short
        fn = getattr(self.stage_label, "set_max_lines", None)
        if fn is not None:
            fn(1 if short else 0)
        h = PANEL_PROGRESS_H_SHORT if short else PANEL_PROGRESS_H
        pb = self.progress_bar
        pb.setStyleSheet(progress_bar_style(PANEL_GOLD, h))
        pb.setFixedHeight(h)
        # an 8 px bar has no room for text: the epoch is in the stage line
        pb.setTextVisible(bool(pb.format() != "%p%") and not short)
        fn = self.on_content_changed
        if callable(fn):
            fn()

    def set_bar_text(self, text: str) -> None:
        """The bar's own words while a round runs (``412 / 1000``); none ->
        no text."""
        pb = self.progress_bar
        if text:
            pb.setFormat(text)
            pb.setTextVisible(not self._short)
        else:
            pb.setFormat("%p%")
            pb.setTextVisible(False)

    def content_height(self, width: int = 0) -> int:
        """The height the foot's content needs at *width*.  ``sizeHint`` is
        wrong for the wrapped stage line (a word-wrapped label's hint is a
        taller, squarer box than it takes at the real width), which left
        blank bands above and below the line."""
        lay = self.layout()
        w = int(width) if width and width > 0 else self.width()
        if lay is not None and w > 0 and lay.hasHeightForWidth():
            return max(int(lay.totalHeightForWidth(w)),
                       self.minimumSizeHint().height())
        return self.sizeHint().height()

    def set_stage_line(self, text: str) -> None:
        new = text or ""
        if self.stage_label.text() == new:
            return
        self.stage_label.setText(new)
        self.stage_label.setToolTip(new if self._short else "")
        fn = self.on_content_changed
        if callable(fn):
            fn()

    def set_running(self, running: bool) -> None:
        """Convenience for tests and the hook: Train greys, Cancel lights."""
        self.train_btn.setEnabled(not running)
        self.train_cancel_btn.setEnabled(running)

    def open_for_round(self, caption: str = "") -> None:
        self.loss.open_for_round(caption)

    def close_idle(self) -> None:
        self.loss.close_idle()


def build_training_foot(
    train_btn,
    train_cancel_btn,
    progress_bar,
    stage_label: QLabel,
    loss_plot,
    loss_view_combo: Optional[QWidget] = None,
    *,
    parent: Optional[QWidget] = None,
) -> TrainingFoot:
    """The pinned foot.  Train is the one primary (solid gold); Cancel is the
    secondary and starts disabled.  Turns round grading on in the plot."""
    foot = TrainingFoot(parent)
    # Step 3's feet carry a rule above their buttons
    foot.setObjectName(FOOT_OBJECT)
    foot.setAttribute(Qt.WA_StyledBackground, True)
    foot.setStyleSheet(panel_foot_style(FOOT_OBJECT))
    lay = QVBoxLayout(foot)
    lay.setContentsMargins(0, 4, 0, 0)
    lay.setSpacing(4)
    row = QHBoxLayout()
    row.setSpacing(6)
    train_btn.setParent(foot)
    train_cancel_btn.setParent(foot)
    train_btn.setText(TRAIN_BTN_TEXT)
    train_cancel_btn.setText(CANCEL_BTN_TEXT)
    # the one primary is Step 3's translucent amber, Cancel the secondary,
    # each with its glyph; the two share the primary's height (D-A, D-K)
    train_btn.setStyleSheet(PRIMARY_BTN_STYLE)
    train_cancel_btn.setStyleSheet(SECONDARY_BTN_STYLE)
    apply_button_icon(train_btn, "mdi6.brain")
    apply_button_icon(train_cancel_btn, "mdi6.stop-circle-outline")
    _h = train_btn.sizeHint().height()
    train_btn.setFixedHeight(_h)
    train_cancel_btn.setFixedHeight(_h)
    train_cancel_btn.setEnabled(False)
    row.addWidget(train_btn, 2)
    row.addWidget(train_cancel_btn, 1)
    lay.addLayout(row)
    progress_bar.setParent(foot)
    progress_bar.setTextVisible(False)
    progress_bar.setFixedHeight(PROGRESS_H)
    lay.addWidget(progress_bar)
    stage_label.setParent(foot)
    if hasattr(stage_label, "set_max_lines"):
        stage_label.set_max_lines(0)        # wrapped until the dock is short
    else:
        stage_label.setWordWrap(True)
    stage_label.setStyleSheet(MONO_LABEL_STYLE)
    stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    stage_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    lay.addWidget(stage_label)
    fn = getattr(loss_plot, "set_grade_by_round", None)
    if callable(fn):
        fn(True)
    fn = getattr(loss_plot, "set_stats_hidden", None)
    if callable(fn):
        fn(True)       # the Loss header's caption says it; not twice
    foot.loss = LossCollapsible(loss_plot, loss_view_combo, foot)
    lay.addWidget(foot.loss)
    foot.train_btn = train_btn
    foot.train_cancel_btn = train_cancel_btn
    foot.progress_bar = progress_bar
    foot.stage_label = stage_label
    foot.loss_plot = loss_plot
    return foot


# -- the panel -----------------------------------------------------------------


class TrainingPanel(QWidget):
    """Lock line (first row, never scrolls), the two columns in scroll areas
    on a horizontal splitter, over the pinned foot on a vertical splitter."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._dragged = False
        self._programmatic = False
        self._run = None
        self._short = False

    def set_short_mode(self, short: bool) -> None:
        """Below the tall-mode height both column descriptions are header
        tooltips (the columns take the room), the stage sentence is one
        elided line and the bar 8 px; the columns snap to a block boundary
        like Curation's right column."""
        self._short = bool(short)
        for col in (getattr(self, "staged_column", None),
                    getattr(self, "setup_column", None)):
            if col is not None and hasattr(col, "title_label"):
                _col_set_short(col, short)
        self.foot.set_short(short)
        self.refit_foot()

    def snap_columns(self) -> None:
        """Snap both column scroll areas' viewports to a block boundary (short
        mode only); the splitter pane they sit in is the room."""
        try:
            holder = getattr(self, "columns_holder", None)
            room = holder.height() if (self._short and holder is not None) \
                else 0
            for sc in (getattr(self, "left_scroll", None),
                       getattr(self, "right_scroll", None)):
                if sc is not None:
                    sc.snap_viewport(room)
        except RuntimeError:          # deleted before the timer fired
            pass

    def _snap_later(self) -> None:
        QTimer.singleShot(0, self.snap_columns)

    def showEvent(self, event):  # noqa: N802 (Qt API)
        super().showEvent(event)
        self._snap_later()

    # splitter sizing (Step 3's _apply_mr_split_sizes)
    def _on_moved(self, *_a) -> None:
        if not self._programmatic:
            self._dragged = True
        self._snap_later()

    def apply_split_sizes(self, opened: bool) -> None:
        sp = self.splitter
        # the room is the shell scroll viewport's, not the splitter's own
        # height (which is max(viewport, minimum hint) inside the scroll area)
        room = sp.height()
        vp = enclosing_viewport_height(self)
        if vp > 0:
            room = min(room, vp)
        total = room - sp.handleWidth()
        if total <= 0:
            return
        floor = self.foot.minimumSizeHint().height()
        bottom = (int(total * SPLIT_OPEN_FRAC) if opened
                  else self.foot.content_height(sp.width()))
        bottom = max(bottom, floor)
        self._programmatic = True
        try:
            sp.setSizes([max(total - bottom, 0), bottom])
        finally:
            self._programmatic = False

    def refit_foot(self) -> None:
        """The foot's content changed (the stage line, the width): re-fit it
        unless the user dragged the splitter or the Loss plot is open."""
        self._snap_later()
        if self._dragged or self.foot.loss.is_open():
            return
        self.apply_split_sizes(False)

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self.refit_foot()

    def _on_loss_toggled(self, opened: bool) -> None:
        if opened and self._dragged:
            return                      # the user's sizes stand
        self.apply_split_sizes(opened)

    # the hooks the main widget forwards (plus the epoch line)
    def on_training_locked(self, locked: bool, round_name: str = "",
                           bucket: str = "") -> None:
        self.lock_line.set(locked, round_name, bucket)
        sc = getattr(self, "setup_column", None)
        if sc is not None and hasattr(sc, "set_locked"):
            sc.set_locked(locked)
        self._snap_later()
        if not locked:
            self.foot.close_idle()
            self.foot.set_running(False)
            self.foot.set_bar_text("")
            self._run = None

    def on_train_started(self, bucket: str, round_name: str) -> None:
        n = round_from_name(round_name)
        fam = "lat" if bucket == "lateral" else "ax"
        m = re.match(r"(.*)_(?:lateral|axial)_round_\d+\s*$", str(round_name))
        self._run = (m.group(1) if m else "", bucket, n)
        self.foot.set_running(True)
        self.foot.set_stage_line(compose_training_stage_line(
            run=self._run[0], bucket=bucket, round_number=n))
        self.foot.open_for_round(
            "" if n is None else f"{fam} r{n} · starting")
        fn = getattr(self.foot.loss_plot, "set_live_family", None)
        if callable(fn):
            fn(bucket)

    def on_epoch(self, epoch: int, epochs: int, loss: Optional[float]) -> None:
        """Seam (d): the running stage line and the Loss header's live
        caption from one epoch message."""
        run, bucket, n = getattr(self, "_run", None) or ("", "lateral", None)
        self.foot.set_stage_line(compose_training_stage_line(
            run=run, bucket=bucket, round_number=n, epoch=epoch,
            epochs=epochs, loss=loss))
        if epochs:
            self.foot.set_bar_text(STAGE_BAR_FORMAT.format(e=epoch, N=epochs))
        fam = "lat" if bucket == "lateral" else "ax"
        self.foot.loss.set_live_caption(LOSS_LIVE_CAPTION.format(
            fam=fam, r=n, e=epoch,
            loss="—" if loss is None else f"{float(loss):.2f}"))

    def on_train_finished(self) -> None:
        """Hook 6: the round's curve is in the plot; back to the idle header
        (the last round) with the Loss closed."""
        self.foot.close_idle()
        self.foot.set_bar_text("")


def build_training_panel(
    setup_column: QWidget,
    staged_column: QWidget,
    foot: TrainingFoot,
    *,
    parent: Optional[QWidget] = None,
) -> TrainingPanel:
    panel = TrainingPanel(parent)
    root = QVBoxLayout(panel)
    root.setContentsMargins(6, 2, 6, 2)     # the panel's 218 px budget
    root.setSpacing(4)
    panel.lock_line = LockLine(panel)
    root.addWidget(panel.lock_line)

    cols = QSplitter(Qt.Horizontal, panel)
    cols.setStyleSheet(SPLITTER_STYLE)
    cols.setHandleWidth(PANEL_SPLIT_HANDLE)
    cols.setChildrenCollapsible(False)
    left = _Scroll(staged_column, cols)
    right = _Scroll(setup_column, cols)
    left.setMinimumWidth(LEFT_MIN_W)
    right.setMinimumWidth(RIGHT_MIN_W)
    cols.addWidget(left)
    cols.addWidget(right)
    cols.setStretchFactor(0, 0)
    cols.setStretchFactor(1, 1)
    cols.setSizes([LEFT_MIN_W + 40, RIGHT_MIN_W + 100])
    cols.setMinimumHeight(SPLIT_TOP_MIN_H)

    sp = CompactSplitter(Qt.Vertical, panel)
    sp.setStyleSheet(SPLITTER_STYLE)
    sp.setHandleWidth(SPLIT_HANDLE)
    sp.setChildrenCollapsible(False)
    # the columns sit in a holder: a block snap caps the scroll areas (and so
    # the columns splitter), and the blank that leaves must stay ABOVE the
    # foot, not inside it -- a splitter pane with a maximum height would hand
    # its leftover to the foot
    holder = QWidget(sp)
    hl = QVBoxLayout(holder)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(0)
    cols.setParent(holder)
    hl.addWidget(cols)
    foot.setParent(sp)
    sp.addWidget(holder)
    sp.addWidget(foot)
    panel.columns_holder = holder
    sp.setStretchFactor(0, 1)
    sp.setStretchFactor(1, 0)
    sp.splitterMoved.connect(panel._on_moved)
    foot.loss.toggled.connect(panel._on_loss_toggled)
    foot.on_content_changed = panel.refit_foot
    root.addWidget(sp, 1)
    panel.columns_splitter = cols
    panel.left_scroll = left
    panel.right_scroll = right
    # the columns' snap measures their content: run it again whenever the
    # holder, a viewport, a column's content or the foot settles
    panel._snap_watch = SnapWatcher(panel, panel.snap_columns)
    panel._snap_watch.watch(holder, left.viewport(), right.viewport(),
                            left.widget(), right.widget(), foot)
    panel.splitter = sp
    panel.foot = foot
    panel.setup_column = setup_column
    panel.staged_column = staged_column
    return panel
