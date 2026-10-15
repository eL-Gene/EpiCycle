"""Renders the delegate's golden rows.

Run as ``python _delegate_golden_render.py OUT_DIR`` in a SUBPROCESS: it
pins the platform (offscreen), the font and the palette, paints one Step 3
narrow row, one Step 3 wide row (a Session row with a tier line) and one
Step 4 row with ``StackRowDelegate.paint`` into a ``QImage`` each and writes
``<name>.png``.  ``test_stack_row_delegate_goldens.py`` compares them
byte-for-byte with the goldens generated at ``533d9cb7`` (before any
delegate key of brick L existed).  Pass ``--extra`` to also render the
rows carrying the NEW keys (the teeth).
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QRect, Qt                               # noqa: E402
from qtpy.QtGui import (QColor, QFont, QImage, QPainter,       # noqa: E402
                        QPixmap, QStandardItem, QStandardItemModel)
from qtpy.QtWidgets import QApplication, QStyleOptionViewItem   # noqa: E402

from napari_epicycle.ui_shell import _stack_row_delegate as D   # noqa: E402

WIDTH = 330


def _thumb(px=128):
    img = QImage(px, px, QImage.Format_RGB32)
    img.fill(QColor("#335577"))
    p = QPainter(img)
    p.fillRect(QRect(px // 4, px // 4, px // 2, px // 2), QColor("#d4a843"))
    p.end()
    return QPixmap.fromImage(img)


def _rows(extra: bool):
    gold = {"sel_bg": QColor(212, 168, 67, 64), "accent": QColor("#d4a843"),
            "chk_mark": QColor("#1a1a1a"), "warn": QColor("#d4a843")}
    s3_narrow = {
        "name": "WholeScale_patch_1.tiff", "word": "predicted", "dots": "",
        "colour": "#d4a843", "axes": "CZYX", "shape_text": "124 x 240 x 240",
        "vox_text": "0.52 x 0.52 x 0.52 um (from Step 1)",
        "chips": (("made by Cell r2", False), ("not the active model", True)),
        "warns_narrow": ("Needs 2 channels, has 1.",), "info": True,
        "icon": "mdi6.layers-outline", "icon_colour": "#d4a843"}
    s3_wide = {
        "name": "WholeScale_patch_4.tiff", "word": "reconstructed",
        "dots": "", "colour": "#6abf69", "axes": "CZYX", "n_channels": 2,
        "line2": "CZYX · 124 × 240 × 240 · 2 ch",
        "line3": "0.52 × 0.52 × 0.52 µm (from Step 1)", "menu": True,
        "eye": True,
        "tier_counts": (("good", 31, (106, 191, 105)),
                        ("review", 6, (212, 168, 67)),
                        ("bad", 2, (200, 80, 80)))}
    s4 = {
        "name": "stack_a.tif", "word": "segmented", "dots": "●●",
        "colour": "#6abf69", "axes": "ZYX", "n_channels": 1,
        "spacing_text": " · 0.5 × 0.5 × 1.0 µm"}
    out = [
        ("s3_narrow", D.MODE_NARROW, dict(show_thumb=True, keep_aspect=True,
                                          show_eye=False, show_info=True,
                                          palette=gold, accent="#d4a843"),
         s3_narrow, False),
        ("s3_wide", D.MODE_WIDE, dict(show_tick=False, keep_aspect=True,
                                      palette=gold, accent="#d4a843"),
         s3_wide, False),
        ("s4_wide", D.MODE_WIDE, dict(keep_aspect=True,
                                      wide_line2="{axes} · {c} ch{spacing}"),
         s4, True),
    ]
    if extra:
        # teeth: the SAME constructor kwargs as the goldens, the SAME record
        # plus the new keys; "_b" changes one visual detail (the strip colour)
        wide_kw = dict(show_tick=False, keep_aspect=True, palette=gold,
                       accent="#d4a843")
        new_keys = dict(s3_wide, strip=((0.5, "#6abf69"), (0.5, "#333333")),
                        status_counts=(("mdi6.check-circle-outline",
                                        "#6abf69", 3, "accepted"),),
                        thumb_px=56)
        out.append(("new_keys_wide", D.MODE_WIDE, dict(wide_kw), new_keys,
                    False))
        out.append(("new_keys_wide_b", D.MODE_WIDE, dict(wide_kw),
                    dict(new_keys, strip=((0.5, "#c85050"), (0.5, "#333333"))),
                    False))
        narrow_kw = dict(show_thumb=True, keep_aspect=True, show_eye=False,
                         show_info=True, palette=gold, accent="#d4a843")
        out.append(("new_keys_narrow", D.MODE_NARROW, dict(narrow_kw),
                    dict(s3_narrow, no_icon=True,
                         line2_plain="CZYX · 124 × 240 × 240"), False))
    return out


def render(out_dir: str, extra: bool = False) -> None:
    app = QApplication.instance() or QApplication([])
    base = QFont("Segoe UI")
    base.setPixelSize(12)
    app.setFont(base)
    for name, mode, kw, rec, checked in _rows(extra):
        deleg = D.StackRowDelegate(mode, **kw)
        model = QStandardItemModel()
        item = QStandardItem(rec["name"])
        item.setData(rec, D.ROLE_RECORD)
        item.setData(_thumb(), D.ROLE_THUMB)
        if "tick" in deleg.layout(QRect(0, 0, WIDTH, 0), rec, base):
            item.setCheckable(True)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        model.appendRow(item)
        idx = model.index(0, 0)
        opt = QStyleOptionViewItem()
        opt.font = base
        opt.rect = QRect(0, 0, WIDTH, 0)
        h = deleg.sizeHint(opt, idx).height()
        opt.rect = QRect(0, 0, WIDTH, h)
        opt.state = opt.state | D.QStyle.State_Enabled
        img = QImage(WIDTH, h, QImage.Format_ARGB32)
        img.fill(QColor("#262930"))
        p = QPainter(img)
        deleg.paint(p, opt, idx)
        p.end()
        img.save(os.path.join(out_dir, name + ".png"), "PNG")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    os.makedirs(args[0], exist_ok=True)
    render(args[0], extra="--extra" in sys.argv)
