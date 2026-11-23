"""The prediction window (``PredictionPreviewDialog``) on its own.

* Its words are constants (``PRED_TEXTS``); it does no file I/O, connects
  no lambda or partial, parents every widget and deletes itself on close.
* It opens on XY + Probability; a Pass or Map choice hands ``on_choice``
  the pair and says it is loading (the dropdowns stay live); the slider
  follows the pass (its range, its axis word, the middle on a Pass change)
  and keeps its place on a Map change; the left pane shows the same plane.
* Every picture reaches Qt contiguous, with its row stride, copied (an odd
  width of 243 is compared pixel by pixel); slider moves are drawn once per
  event-loop turn.
* "Model details…" sits beside Close (hidden with no made-by record) and
  calls ``on_details``; it fits the 566 px dock.

Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _prediction_preview as PP  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_pr_text as T  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_DLG = _REPO / "napari_epicycle" / "fusion_reconstruction" / \
    "_prediction_preview_dialog.py"


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pump(app, n=3):
    from qtpy.QtCore import QCoreApplication, QEvent
    for _ in range(n):
        app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class Calls:
    __slots__ = ("choices", "details", "__weakref__")

    def __init__(self):
        self.choices, self.details = [], 0

    def choice(self, pass_, kind):
        self.choices.append((pass_, kind))

    def on_details(self):
        self.details += 1


def _payload(shape=(5, 9, 7), mode="seg"):
    z, y, x = shape
    seg = (np.arange(z * y * x) % 251).astype(np.uint8).reshape(shape)
    image = {"mode": mode, "seg": seg,
             "helper": seg[::-1].copy() if mode == "pair" else None,
             "shape": shape, "codes": (1, 0)}
    the_map = {"pass": "xy", "kind": "probs", "shape": shape,
               "prob": seg.copy()}
    return {"lines": {"line1": "default · cyto3 (built-in) · predicted x",
                      "line2": "channel to segment 1: red · helper 0: none"},
            "image": image, "caption": PP.left_caption(image),
            "map": the_map, "shape": shape, "shape_changed": False}


def make(qapp, details=True):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction._prediction_preview_dialog \
        import PredictionPreviewDialog
    host = QWidget()
    calls = Calls()
    dlg = PredictionPreviewDialog(
        host, title=T.PRED_TITLE.format(patch="p.tif"), texts=T.PRED_TEXTS,
        on_choice=calls.choice, on_details=calls.on_details,
        details=details)
    return host, dlg, calls


def test_the_window_words_are_constants():
    for key in ("window_title", "title", "line1", "line2",
                "line2_one_channel", "channel_blank", "line_no_record",
                "pass_label", "left_no_record",
                "pass_xy", "pass_xz", "pass_yz", "map_label", "map_prob",
                "map_flows", "left", "right", "left_caption",
                "left_gray_avg", "left_one_channel", "left_contrast",
                "slider", "reading", "reading_flows", "loading",
                "prob_key", "flows_key",
                "loading_map", "failed", "shape_changed", "helper_unused",
                "details", "close"):
        assert key in T.PRED_TEXTS, key
    for key, value in T.PRED_TEXTS.items():
        assert any(value is v for k, v in vars(T).items()
                   if k.isupper() and k != "PRED_TEXTS"), key
    assert T.PRED_TEXTS["details"] is T.S3_MODEL_DETAILS


def test_the_dialog_does_no_file_io():
    tree = ast.parse(_DLG.read_text(encoding="utf-8"))
    calls = {getattr(n.func, "attr", getattr(n.func, "id", ""))
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for io in ("open", "load", "imread", "read_text", "write_text", "save",
               "read_header", "load_map", "load_open", "load_image"):
        assert io not in calls, io


def test_no_lambda_or_partial_is_connected():
    tree = ast.parse(_DLG.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "connect"):
            for arg in node.args:
                assert not isinstance(arg, ast.Lambda), ast.dump(node)
                assert not (isinstance(arg, ast.Call) and getattr(
                    arg.func, "id", getattr(arg.func, "attr", ""))
                    == "partial"), ast.dump(node)
    assert not any(isinstance(n, ast.Lambda) for n in ast.walk(tree))


def test_every_widget_is_parented_and_close_deletes(qapp):
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtWidgets import QApplication
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    before = QApplication.topLevelWidgets()
    host, dlg, _ = make(qapp)
    try:
        from qtpy.QtWidgets import QComboBox
        # a combo box's popup is a top-level window Qt makes and parents to
        # the combo itself; anything else new must be the host or the window
        new = [w for w in QApplication.topLevelWidgets()
               if not any(w is b for b in before)
               and not (isinstance(w.parentWidget(), QComboBox)
                        and dlg.isAncestorOf(w.parentWidget()))]
        assert {id(w) for w in new} == {id(host), id(dlg)}, \
            [type(w).__name__ for w in new]
        assert dlg.parent() is host
        assert dlg.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        destroyed = []
        dlg.destroyed.connect(destroyed.append)
        dlg.show()
        _pump(qapp)
        dlg.close_btn.click()
        _pump(qapp)
        assert destroyed
    finally:
        host.deleteLater()
        _pump(qapp)


def test_it_opens_on_xy_and_probability_and_says_loading(qapp):
    host, dlg, calls = make(qapp)
    try:
        assert dlg.windowTitle() == "Prediction — p.tif"
        assert dlg.current_choice() == ("xy", "probs")
        assert dlg.pass_combo.currentText() == T.PRED_PASS_XY
        assert dlg.map_combo.currentText() == T.PRED_MAP_PROB
        assert dlg.left_box.text() == T.PRED_LOADING
        assert dlg.right_box.text() == T.PRED_LOADING
        assert dlg.left_head.text() == T.PRED_LEFT
        assert dlg.right_head.text() == "Probability, XY pass"
        assert dlg.reading_label.text() == T.PRED_READING
        assert dlg.left_note.text() == T.PRED_LEFT_CONTRAST
        assert dlg.details_btn.text() == T.S3_MODEL_DETAILS
        assert dlg.close_btn.text() == T.PRED_CLOSE
        assert calls.choices == []
    finally:
        host.deleteLater()
        _pump(qapp)


def test_the_first_answer_fills_both_panes_and_the_slider(qapp):
    host, dlg, _ = make(qapp)
    try:
        dlg.set_image(_payload())
        assert dlg.line1_label.text().startswith("default · cyto3")
        assert dlg.line2_label.text().startswith("channel to segment 1: red")
        assert dlg.left_caption.text() == "1: red"
        assert dlg.slider.maximum() == 4 and dlg.slider.value() == 2
        assert dlg.slider_label.text() == "z 2 / 4"      # from 0
        assert dlg.left_box.pixmap() is not None
        assert dlg.last_left.shape == (9, 7, 3)
        assert dlg.last_right.shape == (9, 7, 3)
    finally:
        host.deleteLater()
        _pump(qapp)


def test_a_pass_change_moves_the_slider_and_asks_for_that_file(qapp):
    host, dlg, calls = make(qapp)
    try:
        dlg.set_image(_payload())
        dlg.slider.setValue(4)
        _pump(qapp)
        dlg.pass_combo.setCurrentIndex(1)              # XZ
        assert calls.choices == [("xz", "probs")]
        assert dlg.slider.maximum() == 8 and dlg.slider.value() == 4
        assert dlg.slider_label.text() == "y 4 / 8"
        assert dlg.right_box.text() == T.PRED_LOADING_MAP.format(
            **{"pass": "XZ"})
        assert dlg.right_head.text() == "Probability, XZ pass"
        assert dlg.last_left.shape == (5, 7, 3)        # the same plane
        assert dlg.pass_combo.isEnabled() and dlg.map_combo.isEnabled()
        # the map arrives: the right pane shows the same XZ plane
        the_map = dict(_payload()["map"], **{"pass": "xz"})
        dlg.set_map(the_map)
        assert dlg.last_right.shape == (5, 7, 3)
        dlg.pass_combo.setCurrentIndex(2)              # YZ
        assert calls.choices[-1] == ("yz", "probs")
        assert dlg.slider.maximum() == 6 and dlg.slider_label.text() == \
            "x 3 / 6"
        assert dlg.last_left.shape == (9, 5, 3)        # Y down, Z across
    finally:
        host.deleteLater()
        _pump(qapp)


def test_a_map_change_keeps_the_plane_and_shows_the_flows(qapp):
    host, dlg, calls = make(qapp)
    try:
        dlg.set_image(_payload())
        dlg.slider.setValue(1)
        _pump(qapp)
        dlg.map_combo.setCurrentIndex(1)               # Flows
        assert calls.choices == [("xy", "flows")]
        assert dlg.slider.value() == 1
        assert dlg.right_head.text() == "Flows, XY pass"
        assert dlg.reading_label.text() == T.PRED_READING_FLOWS
        shape = (5, 9, 7)
        rng = np.random.default_rng(1)
        dlg.set_map({"pass": "xy", "kind": "flows", "shape": shape,
                     "comps": rng.normal(0, 1, (2,) + shape).astype(
                         np.float16), "lo": 0.1, "hi": 2.0})
        assert dlg.last_right.shape == (9, 7, 3)
        assert dlg.slider.value() == 1
    finally:
        host.deleteLater()
        _pump(qapp)


def test_failures_are_said_in_the_panes(qapp):
    host, dlg, _ = make(qapp)
    try:
        sentence = PP.too_large_sentence((200, 1024, 1024))
        dlg.set_failed(sentence)
        assert dlg.left_box.text() == dlg.right_box.text() == sentence
        assert not dlg.pass_combo.isEnabled()
        assert not dlg.map_combo.isEnabled()
        host2, dlg2, _ = make(qapp)
        dlg2.set_image(_payload())
        dlg2.set_map_failed(T.PRED_FAILED.format(error="OSError: x"))
        assert dlg2.right_box.text().startswith("Could not read this")
        host2.deleteLater()
    finally:
        host.deleteLater()
        _pump(qapp)


def test_a_shape_change_shows_the_probability_only(qapp):
    host, dlg, _ = make(qapp)
    try:
        payload = _payload()
        payload.update(image=None, caption="", shape_changed=True)
        dlg.set_image(payload)
        assert dlg.left_box.text() == T.S3_PRED_SHAPE_CHANGED
        assert dlg.last_left is None and dlg.last_right is not None
    finally:
        host.deleteLater()
        _pump(qapp)


def test_a_big_plane_is_thinned_before_it_is_coloured(qapp):
    """A 2048 x 2048 plane is coloured at about twice the pane, so a slider
    move costs the same at any size."""
    host, dlg, _ = make(qapp)
    try:
        dlg.set_image(_payload(shape=(3, 2048, 2048)))
        assert max(dlg.last_left.shape[:2]) <= 2 * 248
        assert max(dlg.last_right.shape[:2]) <= 2 * 248
    finally:
        host.deleteLater()
        _pump(qapp)


def test_closing_drops_the_arrays(qapp):
    host, dlg, _ = make(qapp)
    try:
        dlg.set_image(_payload())
        dlg.show()
        _pump(qapp)
        dlg.close()
        assert dlg._image is None and dlg._map is None
        assert dlg.last_left is None and dlg.last_right is None
    finally:
        host.deleteLater()
        _pump(qapp)


def test_the_picture_keeps_its_rows_at_an_odd_width(qapp):
    from napari_epicycle.fusion_reconstruction._prediction_preview_dialog \
        import to_pixmap
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(31, 243, 3)).astype(np.uint8)
    img = to_pixmap(rgb[:, :, :]).toImage()
    assert (img.width(), img.height()) == (243, 31)
    for r in range(31):
        c = (r * 37) % 243
        col = img.pixelColor(c, r)
        assert (col.red(), col.green(), col.blue()) == tuple(rgb[r, c]), r
    # a non-contiguous view is copied first
    view = np.asfortranarray(rgb)
    img2 = to_pixmap(view).toImage()
    col = img2.pixelColor(242, 30)
    assert (col.red(), col.green(), col.blue()) == tuple(rgb[30, 242])


def test_slider_moves_are_drawn_once_per_turn(qapp, monkeypatch):
    from napari_epicycle.fusion_reconstruction import (
        _prediction_preview_dialog as D)
    host, dlg, _ = make(qapp)
    try:
        dlg.set_image(_payload())
        n = []
        real = D.PP.render_image

        def counting(*a, **k):
            n.append(1)
            return real(*a, **k)

        monkeypatch.setattr(D.PP, "render_image", counting)
        for v in (0, 1, 2, 3, 4, 3):
            dlg.slider.setValue(v)
        assert n == []                       # nothing drawn inside the drag
        _pump(qapp)
        assert len(n) == 1
    finally:
        host.deleteLater()
        _pump(qapp)


def test_model_details_beside_close(qapp):
    host, dlg, calls = make(qapp)
    try:
        dlg.show()
        _pump(qapp)
        assert dlg.details_btn.isVisible()
        dlg.details_btn.click()
        assert calls.details == 1
    finally:
        dlg.close()
        host.deleteLater()
        _pump(qapp)
    host, dlg, calls = make(qapp, details=False)
    try:
        dlg.show()
        _pump(qapp)
        assert not dlg.details_btn.isVisible()
    finally:
        dlg.close()
        host.deleteLater()
        _pump(qapp)


def test_the_window_fits_the_dock(qapp):
    host, dlg, _ = make(qapp)
    try:
        dlg.set_image(_payload())
        dlg.show()
        _pump(qapp, 6)
        assert dlg.width() <= 560, dlg.width()
        assert dlg.minimumSizeHint().width() <= 566
        screen = dlg.screen().availableGeometry()
        assert dlg.height() <= screen.height()
    finally:
        dlg.close()
        host.deleteLater()
        _pump(qapp)


def test_the_right_pane_carries_its_key(qapp):
    """The probability's fixed scale under it; the flows' colour key under
    them."""
    host, dlg, _ = make(qapp)
    try:
        dlg.set_image(_payload())
        assert dlg.right_caption.text() == T.PRED_PROB_KEY == "−6 … 6"
        dlg.map_combo.setCurrentIndex(1)
        assert dlg.right_caption.text() == T.PRED_FLOWS_KEY == \
            "hue = direction, brightness = strength"
        dlg.map_combo.setCurrentIndex(0)
        assert dlg.right_caption.text() == T.PRED_PROB_KEY
    finally:
        dlg.close()
        host.deleteLater()
        _pump(qapp)


def _stacked(parts):
    """Each part starts at or below the end of the one above it."""
    bad = []
    for above, below in zip(parts, parts[1:]):
        if below.isHidden() or above.isHidden():
            continue
        if below.geometry().top() < above.geometry().bottom():
            bad.append((above.objectName() or type(above).__name__,
                        above.geometry().bottom(), below.geometry().top()))
    return bad


@pytest.mark.parametrize("kind", ["probs", "flows"])
@pytest.mark.parametrize("two_line_provenance", [False, True])
def test_nothing_overlaps_on_either_map(qapp, kind, two_line_provenance):
    """The two-line flows reading (and a provenance that wraps) never
    squeezes the panes: each caption sits under its picture, the reading
    has the room its text needs at the window's width, all by the
    window's own size rule."""
    host, dlg, _ = make(qapp)
    try:
        payload = _payload(mode="pair")
        if two_line_provenance:
            payload["lines"]["line1"] = (
                "wound-v2-membrane-nuclei-refined-2026-10-04 · "
                "wound-v2-membrane-nuclei-refined-2026-10-04-from-step2 · "
                "predicted 2026-10-26 23:14")
        dlg.show()
        _pump(qapp, 4)
        dlg.set_image(payload)
        if kind == "flows":
            dlg.map_combo.setCurrentIndex(1)
        _pump(qapp, 6)
        assert _stacked([dlg.left_head, dlg.left_box, dlg.left_caption,
                         dlg.left_note]) == []
        assert _stacked([dlg.right_head, dlg.right_box,
                         dlg.right_caption]) == []
        r = dlg.reading_label
        assert r.height() >= r.heightForWidth(r.width()), (
            r.height(), r.heightForWidth(r.width()))
        holder_bottom = dlg.left_note.mapTo(
            dlg, dlg.left_note.rect().bottomLeft()).y()
        assert holder_bottom <= dlg.slider.geometry().top()
        assert r.geometry().top() >= dlg.slider.geometry().bottom()
        assert dlg.width() <= 560
    finally:
        dlg.close()
        host.deleteLater()
        _pump(qapp)


def test_the_magma_table_runs_dark_to_light():
    from napari_epicycle.fusion_reconstruction._prediction_preview_dialog \
        import magma_lut
    lut = magma_lut()
    assert lut.shape == (256, 3) and lut.dtype == np.uint8
    assert int(lut[0].sum()) < 30 and int(lut[255].sum()) > 600
