"""Capture helpers for the three R1 lift goldens.

The goldens are captured ONCE from the untouched Step 4 code and never
regenerated to make a test pass.  ``test_step4_lift_goldens.py`` compares
the live panel / delegate / text tables with them; writing a golden needs
``STEP4_LIFT_GOLDEN_WRITE=1`` AND the file missing.

* ``dump_panel_tree(w)`` -- the whole ``SegmentStack`` widget tree (the
  Export window ``Step4ExportDialog`` is its child), per widget in tree
  order: class, attribute names, objectName, text, toolTip, styleSheet,
  enabled / hidden / checked, spin values / ranges / steps / decimals /
  special text / suffix / prefix, combo items with data, list items, every
  collapsible's title, engine label and expanded state; plus the
  ``_adv_engine_labels`` texts and the shared style constants.
* ``dump_delegate()`` -- ``sizeHint`` and a hash of the painted ``QImage``
  for fixed records in the three modes, and the tooltip per painted part
  (run in a real-font subprocess by the test).
* ``dump_panel_layout(w)`` -- the fourth golden (added after the R1
  audits, captured from the pre-lift commit ``db8a0917``): size policies,
  set minimum / maximum sizes, cursors, mouse transparency, label wrap /
  format / alignment / buddy, and every layout's margins, spacing and
  items in order.
* ``dump_text_names()`` -- every name of ``_step4_text`` and
  ``_step4_models`` with its value (JSON-normalised; callables by name),
  compared as dicts by EQUALITY.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
PANEL_GOLDEN = GOLDEN_DIR / "step4_panel_tree.json"
DELEGATE_GOLDEN = GOLDEN_DIR / "step4_row_delegate.json"
TEXT_GOLDEN = GOLDEN_DIR / "step4_text_names.json"
#: the fourth golden (an audit of R1): layout and interaction properties
#: the panel tree golden does not record; see ``dump_panel_layout``
LAYOUT_GOLDEN = GOLDEN_DIR / "step4_panel_layout.json"


# ---------------------------------------------------------------- helpers

def _norm(value, subs):
    """JSON-normalised *value* with every workspace path replaced."""
    def fix(s):
        for old, new in subs:
            s = s.replace(old, new)
        return s

    if isinstance(value, str):
        return fix(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(_norm(k, subs)): _norm(v, subs) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_norm(v, subs) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_norm(v, subs) for v in value)
    return fix(str(value))


def path_subs(ws) -> list:
    if ws is None:
        return []
    s = str(ws)
    alts = {s, s.replace("\\", "/"), s.replace("/", "\\")}
    try:
        r = str(Path(ws).resolve())
        alts |= {r, r.replace("\\", "/"), r.replace("/", "\\")}
    except OSError:
        pass
    return [(a, "<WS>") for a in sorted(alts, key=len, reverse=True)]


def _ev(x):
    """A Qt enum / flag as an int (PyQt6 enums have ``.value``)."""
    return int(getattr(x, "value", x))


# ---------------------------------------------------------------- the panel

def _widget_entry(wid, attrs, subs):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QAbstractButton, QAbstractSpinBox, QComboBox, QDialog,
        QDoubleSpinBox, QLabel, QLineEdit, QListWidget, QProgressBar,
        QSpinBox, QSplitter,
    )
    from superqt import QCollapsible

    e = {"class": type(wid).__name__,
         "attrs": sorted(attrs.get(id(wid), ())),
         "objectName": wid.objectName(),
         "toolTip": wid.toolTip(),
         "styleSheet": wid.styleSheet(),
         "enabled": wid.isEnabled(),
         "hidden": wid.isHidden()}
    if isinstance(wid, (QLabel, QAbstractButton, QLineEdit)):
        e["text"] = wid.text()
        if "gpu_status_label" in e["attrs"]:
            e["text"] = "<the machine's GPU line>"     # Diagnostics
    if isinstance(wid, QLineEdit):
        e["placeholder"] = wid.placeholderText()
        e["readOnly"] = wid.isReadOnly()
    if isinstance(wid, QAbstractButton):
        e["checkable"] = wid.isCheckable()
        e["checked"] = wid.isChecked()
    if isinstance(wid, QAbstractSpinBox):
        e["specialValueText"] = wid.specialValueText()
    if isinstance(wid, (QSpinBox, QDoubleSpinBox)):
        e["value"] = wid.value()
        e["range"] = [wid.minimum(), wid.maximum()]
        e["step"] = wid.singleStep()
        e["suffix"] = wid.suffix()
        e["prefix"] = wid.prefix()
        if isinstance(wid, QDoubleSpinBox):
            e["decimals"] = wid.decimals()
    if isinstance(wid, QComboBox):
        e["items"] = [[wid.itemText(i), wid.itemData(i),
                       wid.itemData(i, Qt.ToolTipRole)]
                      for i in range(wid.count())]
        e["currentIndex"] = wid.currentIndex()
    if isinstance(wid, QListWidget):
        from napari_epicycle.segmentation._stack_row_delegate import (
            ROLE_NAME, ROLE_RECORD, ROLE_STATUS,
        )
        items = []
        for i in range(wid.count()):
            it = wid.item(i)
            items.append({"text": it.text(), "toolTip": it.toolTip(),
                          "check": _ev(it.checkState()),
                          "flags": _ev(it.flags()),
                          "name": it.data(ROLE_NAME),
                          "status": it.data(ROLE_STATUS),
                          "record": it.data(ROLE_RECORD)})
        e["items"] = items
    if isinstance(wid, QProgressBar):
        e["progress"] = [wid.minimum(), wid.maximum(), wid.value(),
                         wid.format(), wid.isTextVisible()]
    if isinstance(wid, QSplitter):
        e["splitter"] = [_ev(wid.orientation()), wid.handleWidth(),
                         wid.childrenCollapsible()]
    if isinstance(wid, QCollapsible):
        eng = getattr(wid, "_epicycle_engine_label", None)
        e["collapsible"] = {
            "title": wid.text(), "expanded": wid.isExpanded(),
            "engine": eng.text() if eng is not None else None,
            "engineObject": eng.objectName() if eng is not None else None,
            "engineStyle": eng.styleSheet() if eng is not None else None,
            "symbol": getattr(wid, "_epicycle_collapsed_symbol", None)}
    if isinstance(wid, QDialog):
        e["windowTitle"] = wid.windowTitle()
    return _norm(e, subs)


def _walk(wid, out, attrs, subs):
    from qtpy.QtWidgets import QWidget
    out.append(_widget_entry(wid, attrs, subs))
    for ch in wid.children():
        if isinstance(ch, QWidget):
            _walk(ch, out, attrs, subs)


def dump_panel_tree(w, ws=None) -> dict:
    """Everything the panel golden holds, for the panel *w*."""
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.segmentation import ui as U
    from napari_epicycle.segmentation._stack_row_delegate import (
        CHECK_DISABLED_STYLE,
    )

    subs = path_subs(ws)
    attrs = {}
    for name, val in vars(w).items():
        if isinstance(val, QWidget):
            attrs.setdefault(id(val), []).append(name)
    tree = []
    _walk(w, tree, attrs, subs)
    return {
        "tree": tree,
        "adv_engine_labels": _norm({k: v.text() for k, v in
                                    w._adv_engine_labels.items()}, subs),
        "constants": {
            "LOCK_LINE_STYLE": U.LOCK_LINE_STYLE,
            "ENGINE_NAME_STYLE": U.ENGINE_NAME_STYLE,
            "FORM_FIELD_W": U.FORM_FIELD_W,
            "CHECK_DISABLED_STYLE": CHECK_DISABLED_STYLE,
            "ZWSP": U.ZWSP,
        },
    }


def build_panel_states(qapp, tmp_path, dump=None) -> dict:
    """The two states the panel golden is taken in: no workspace, and a
    workspace with stacks, a model and a tick (② open, Advanced open).
    *dump* is the capture (default ``dump_panel_tree``)."""
    if dump is None:
        dump = dump_panel_tree
    from ._step4_panel import (close, make_workspace, panel, pump,
                               refined_bundle, tick)

    out = {}
    w = panel(qapp, None)
    try:
        pump(qapp, 10)
        out["empty"] = dump(w)
    finally:
        close(w)
        pump(qapp, 5)

    ws = make_workspace(Path(tmp_path), patches=("P1.tif", "P2.tif",
                                                  "P3.tif"),
                        one_channel=("P3.tif",))
    refined_bundle(ws, None, "wound-v2", channels=(2, 1))
    w = panel(qapp, ws)
    try:
        c = w.controller
        from napari_epicycle.fusion_reconstruction._run_management import (
            list_cross_run_refined_models,
        )
        from napari_epicycle.segmentation import _step4_models as M
        rows = M.step4_model_rows(list_cross_run_refined_models(
            ws, trained_only=True))
        c._on_model_use(dict(rows[0], _kind="refined"))
        tick(w, qapp, "P1.tif")
        w.accordion.setCurrentIndex(1)
        w._adv_collapsible.expand(animate=False)
        pump(qapp, 10)
        out["workspace"] = dump(w, ws)
    finally:
        close(w)
        pump(qapp, 5)
    return out


# ---------------------------------------------------------------- the layout

_QWIDGETSIZE_MAX = 16777215


def _wid_ident(wid, attrs, subs):
    """A stable name for *wid*: class, objectName, attribute names and the
    first 40 characters of its text (when it has one)."""
    from qtpy.QtWidgets import QAbstractButton, QLabel
    text = ""
    if isinstance(wid, (QLabel, QAbstractButton)):
        text = wid.text()[:40]
    return _norm(f"{type(wid).__name__}#{wid.objectName()}"
                 f"|{','.join(sorted(attrs.get(id(wid), ())))}|{text}", subs)


def _layout_entry(lay, attrs, subs):
    from qtpy.QtWidgets import QBoxLayout, QFormLayout, QGridLayout
    m = lay.contentsMargins()
    e = {"class": type(lay).__name__,
         "objectName": lay.objectName(),
         "margins": [m.left(), m.top(), m.right(), m.bottom()],
         "sizeConstraint": _ev(lay.sizeConstraint())}
    if isinstance(lay, QGridLayout):
        e["spacing"] = [lay.horizontalSpacing(), lay.verticalSpacing()]
        e["columns"] = [[lay.columnStretch(c), lay.columnMinimumWidth(c)]
                        for c in range(lay.columnCount())]
        e["rows"] = [[lay.rowStretch(r), lay.rowMinimumHeight(r)]
                     for r in range(lay.rowCount())]
    elif isinstance(lay, QFormLayout):
        e["spacing"] = [lay.horizontalSpacing(), lay.verticalSpacing()]
    else:
        e["spacing"] = lay.spacing()
    if isinstance(lay, QBoxLayout):
        e["direction"] = _ev(lay.direction())
    items = []
    for i in range(lay.count()):
        it = lay.itemAt(i)
        rec = {"alignment": _ev(it.alignment())}
        if isinstance(lay, QBoxLayout):
            rec["stretch"] = lay.stretch(i)
        elif isinstance(lay, QGridLayout):
            rec["cell"] = list(lay.getItemPosition(i))
        elif isinstance(lay, QFormLayout):
            row, role = lay.getItemPosition(i)
            rec["cell"] = [row, _ev(role)]
        w, sub, sp = it.widget(), it.layout(), it.spacerItem()
        if w is not None:
            rec["widget"] = _wid_ident(w, attrs, subs)
        elif sub is not None:
            rec["layout"] = _layout_entry(sub, attrs, subs)
        elif sp is not None:
            hint = sp.sizeHint()
            pol = sp.sizePolicy()
            rec["spacer"] = [hint.width(), hint.height(),
                             _ev(pol.horizontalPolicy()),
                             _ev(pol.verticalPolicy())]
        items.append(rec)
    e["items"] = items
    return e


def _widget_layout_entry(wid, attrs, subs):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel
    e = {"id": _wid_ident(wid, attrs, subs)}
    pol = wid.sizePolicy()
    e["sizePolicy"] = [_ev(pol.horizontalPolicy()), _ev(pol.verticalPolicy()),
                       pol.horizontalStretch(), pol.verticalStretch()]
    mn, mx = wid.minimumSize(), wid.maximumSize()
    if mn.width() or mn.height():
        e["minimum"] = [mn.width(), mn.height()]
    if mx.width() != _QWIDGETSIZE_MAX or mx.height() != _QWIDGETSIZE_MAX:
        e["maximum"] = [mx.width(), mx.height()]
    if wid.testAttribute(Qt.WA_SetCursor):
        e["cursor"] = _ev(wid.cursor().shape())
    if wid.testAttribute(Qt.WA_TransparentForMouseEvents):
        e["transparentForMouse"] = True
    if isinstance(wid, QLabel):
        e["label"] = {"wordWrap": wid.wordWrap(),
                      "textFormat": _ev(wid.textFormat()),
                      "alignment": _ev(wid.alignment()),
                      "buddy": wid.buddy() is not None}
    lay = wid.layout()
    if lay is not None:
        e["layout"] = _layout_entry(lay, attrs, subs)
    return e


def _walk_layout(wid, out, attrs, subs):
    from qtpy.QtWidgets import QWidget
    out.append(_widget_layout_entry(wid, attrs, subs))
    for ch in wid.children():
        if isinstance(ch, QWidget):
            _walk_layout(ch, out, attrs, subs)


def dump_panel_layout(w, ws=None) -> dict:
    """Per widget in tree order: size policy (h, v, stretches), minimum /
    maximum size where set, cursor where set, ``WA_TransparentForMouse
    Events``, a label's wordWrap / textFormat / alignment / buddy; and per
    layout its margins, spacing (h + v for a grid or form), size
    constraint, grid column / row stretches and minimums, and its items IN
    ORDER with their stretch, cell and alignment."""
    from qtpy.QtWidgets import QWidget
    subs = path_subs(ws)
    attrs = {}
    for name, val in vars(w).items():
        if isinstance(val, QWidget):
            attrs.setdefault(id(val), []).append(name)
    tree = []
    _walk_layout(w, tree, attrs, subs)
    return {"tree": tree}


def build_panel_layout_states(qapp, tmp_path) -> dict:
    """``dump_panel_layout`` in the two states of ``build_panel_states``."""
    return build_panel_states(qapp, tmp_path, dump=dump_panel_layout)


# ---------------------------------------------------------------- the rows

def _records():
    """Fixed row records: every part the delegate paints."""
    base = {"name": "Wound_patch_1.tif", "word": "predicted", "dots": "●●○",
            "colour": "#6abf69", "icon": "mdi6.check-circle-outline",
            "axes": "CZYX", "n_channels": 2,
            "spacing_text": "2.0 × 0.5 × 0.5 µm (from Step 1)",
            "shape_text": "40×512×512 · 2 ch · 1,234 cells",
            "tickable": True}
    return {
        "plain": dict(base),
        "voxels": dict(base, spacing_text="voxels", in_voxels=True,
                       warns_wide=["No voxel size — numbers will be in "
                                   "voxels. Set spacing…"],
                       warns_narrow=["No voxel size — runs in voxels."]),
        "chips": dict(base, chips=[["made by wound-v2", False],
                                   ["made by other — not the active model",
                                    True]]),
        "greyed": dict(base, tickable=False, word="failed",
                       colour="#e07070",
                       warns_narrow=["Its axis order could not be read, so "
                                     "it cannot run."]),
        "long": dict(base, name="A_really_long_stack_name_that_must_elide_"
                     "in_the_middle_of_the_row.tif",
                     shape_text="400×2048×2048 · 4 ch · 123,456 cells and a "
                     "line long enough to wrap"),
    }


def _make_delegate(D, mode, lw):
    """The delegate as Step 4's panel builds it (``ui.py``'s three
    construction sites): after the lift, with its two keywords."""
    import inspect
    params = inspect.signature(D.StackRowDelegate.__init__).parameters
    if "help_lines" in params:
        from napari_epicycle.segmentation import _step4_text as T
        return D.StackRowDelegate(mode, lw, help_lines=T.STEP4_HELP_LINES,
                                  wide_line2=T.ROW_WIDE_LINE2)
    return D.StackRowDelegate(mode, lw)


def dump_delegate() -> dict:
    """``sizeHint``, a painted-image hash and the tooltips, per mode,
    record and variant, at the width the user gets."""
    from qtpy.QtCore import QEvent, QPoint, Qt
    from qtpy.QtGui import QColor, QHelpEvent, QImage, QPainter, QPixmap
    from qtpy.QtWidgets import QListWidgetItem, QStyle, QStyleOptionViewItem

    from napari_epicycle.segmentation import _stack_row_delegate as D

    shown = []

    class _Tip:
        @staticmethod
        def showText(pos, text, view=None):
            shown.append(text)

        @staticmethod
        def hideText():
            shown.append(None)

    real_tip = D.QToolTip
    D.QToolTip = _Tip
    out = {}
    try:
        widths = {D.MODE_WIDE: 566, D.MODE_NARROW: 218, D.MODE_MEASURE: 218}
        for mode, width in widths.items():
            lw = D.StackListWidget()
            dlg = _make_delegate(D, mode, lw)
            lw.setItemDelegate(dlg)
            lw.resize(width, 600)
            lw.show()
            for key, rec in _records().items():
                for variant in ("idle", "ticked", "selected", "locked",
                                "eye_locked", "thumb"):
                    lw.clear()
                    lw.setProperty("step4Locked", variant == "locked")
                    lw.setProperty("step4EyeLocked", variant == "eye_locked")
                    it = QListWidgetItem(rec["name"])
                    it.setData(D.ROLE_NAME, rec["name"])
                    it.setData(D.ROLE_RECORD, rec)
                    flags = it.flags() | Qt.ItemIsUserCheckable
                    if variant == "locked":
                        flags &= ~Qt.ItemIsUserCheckable
                    it.setFlags(flags)
                    # the enum through setData, as the panel writes it
                    # (``setCheckState`` stores a bare int under PyQt6)
                    it.setData(Qt.CheckStateRole,
                               Qt.Checked if variant == "ticked"
                               else Qt.Unchecked)
                    if variant == "thumb":
                        pm = QPixmap(64, 64)
                        pm.fill(QColor(200, 40, 160))
                        it.setData(D.ROLE_THUMB, pm)
                    lw.addItem(it)
                    idx = lw.model().index(0, 0)
                    opt = QStyleOptionViewItem()
                    lw.initViewItemOption(opt)
                    opt.rect = lw.visualItemRect(it)
                    opt.widget = lw
                    opt.state |= QStyle.StateFlag.State_Enabled
                    if variant == "selected":
                        opt.state |= QStyle.StateFlag.State_Selected
                    hint = dlg.sizeHint(opt, idx)
                    img = QImage(hint.width(), hint.height(),
                                 QImage.Format.Format_ARGB32)
                    img.fill(QColor(38, 41, 48))
                    opt.rect.moveTo(0, 0)
                    opt.rect.setSize(hint)
                    p = QPainter(img)
                    dlg.paint(p, opt, idx)
                    p.end()
                    ptr = img.constBits()
                    ptr.setsize(img.sizeInBytes())
                    digest = hashlib.sha1(bytes(ptr)).hexdigest()
                    g = dlg.layout(opt.rect, rec, opt.font)
                    parts = {"tick": g.get("tick"), "eye": g.get("eye"),
                             "menu": g.get("menu"),
                             "l1": g["lines"].get("l1")}
                    if g["chips"]:
                        parts["chip0"] = g["chips"][0][0]
                    tips = {}
                    for part, r in parts.items():
                        if r is None:
                            continue
                        shown.clear()
                        ev = QHelpEvent(QEvent.Type.ToolTip, r.center(),
                                        QPoint(0, 0))
                        dlg.helpEvent(ev, lw, opt, idx)
                        tips[part] = shown[-1] if shown else "<none>"
                    out[f"{mode}/{key}/{variant}"] = {
                        "size": [hint.width(), hint.height()],
                        "sha1": digest, "tips": tips}
            lw.hide()
            lw.deleteLater()
    finally:
        D.QToolTip = real_tip
    return out


# ---------------------------------------------------------------- the text

def _value(v):
    import types
    if isinstance(v, types.ModuleType):
        return None
    if isinstance(v, type):
        return {"__class__": v.__name__}
    if callable(v):
        return {"__callable__": getattr(v, "__name__", repr(v))}
    mod = getattr(type(v), "__module__", "")
    if mod in ("typing", "__future__"):
        return None
    if isinstance(v, (str, int, float, bool, dict, list, tuple, set,
                      frozenset)) or v is None:
        return json.loads(json.dumps(_norm(v, [])))
    return {"__repr__": repr(v)}


def dump_text_names() -> dict:
    from napari_epicycle.segmentation import _step4_models as M
    from napari_epicycle.segmentation import _step4_text as T
    out = {}
    for mod_name, mod in (("_step4_text", T), ("_step4_models", M)):
        names = {}
        for k, v in vars(mod).items():
            if k.startswith("__"):
                continue
            if v is not None and _value(v) is None:
                continue
            names[k] = _value(v)
        out[mod_name] = names
    return out


# ---------------------------------------------------------------- Set spacing

#: the fifth golden: Step 4's
#: ``SetSpacingDialog`` captured from the UNTOUCHED code before it moved
#: into ``ui_shell/_set_spacing_dialog.py``
SPACING_DIALOG_GOLDEN = GOLDEN_DIR / "step4_spacing_dialog.json"


def _spacing_dialog_states():
    """``(key, names, record)`` the golden is taken in: one stack with a
    user's spacing, a flipped Z and an override; three stacks and no
    record; one stack whose record holds values the dialog must refuse."""
    return [
        ("one_user", ["Wound_patch_1.tif"],
         {"spacing": {"zyx": [2.0, 0.5, 0.25], "source": "user"},
          "z_orientation": "basal_up", "axes_override": "ZCYX"}),
        ("three_none", ["P1.tif", "P2.tif", "P3.tif"], None),
        ("one_bad", ["P9.tif"],
         {"spacing": {"zyx": ["x", 0.2, 0.3]}, "z_orientation": "sideways",
          "axes_override": "XYZ"}),
    ]


def dump_spacing_dialog(dlg) -> dict:
    """The dialog's widget tree (``_widget_entry``), its layout tree
    (``_widget_layout_entry``), ``values()``, modality and window title."""
    from qtpy.QtWidgets import QWidget
    attrs = {}
    for name, val in vars(dlg).items():
        if isinstance(val, QWidget):
            attrs.setdefault(id(val), []).append(name)
    tree, layout = [], []
    _walk(dlg, tree, attrs, [])
    _walk_layout(dlg, layout, attrs, [])
    return {"tree": tree, "layout": layout,
            "values": _norm(dlg.values(), []),
            "modal": dlg.isModal(),
            "windowTitle": dlg.windowTitle()}


def build_spacing_dialog_states(qapp) -> dict:
    """``dump_spacing_dialog`` for each state of ``_spacing_dialog_states``,
    on Step 4's ``segmentation._set_spacing_dialog.SetSpacingDialog``."""
    from napari_epicycle.segmentation._set_spacing_dialog import (
        SetSpacingDialog,
    )
    out = {}
    for key, names, record in _spacing_dialog_states():
        dlg = SetSpacingDialog(None, names=names, record=record)
        try:
            out[key] = dump_spacing_dialog(dlg)
        finally:
            dlg.deleteLater()
            qapp.processEvents()
    return json.loads(json.dumps(out))
