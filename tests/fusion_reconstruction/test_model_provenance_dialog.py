"""R6a: the model details window ("On
screen").

* every key of every view is listed GENERICALLY -- a key no code has heard
  of (a foreign view, a foreign key, a foreign nested key) still shows,
  nested dicts as ``a.b``;
* the notes are editable and Save hands them to the caller's ``on_save``
  (the window does no file I/O);
* ``locked=True`` greys Save and writes the reason ON SCREEN (Qt shows no
  tooltip on a disabled widget);
* every widget is parented, no lambda is connected, the window deletes
  itself on close;
* the size fits 566 px of dock, measured in a real-font subprocess.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction import _model_provenance as P
from napari_epicycle.fusion_reconstruction import _step3_pr_text as PT

_REPO = Path(__file__).resolve().parents[2]
_FONT_DIR = "C:/Windows/Fonts"
DOCK_W = 566
NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _pump(app, n=5):
    for _ in range(n):
        app.processEvents()


def sample_block() -> dict:
    """A realistic block plus a foreign key at every level."""
    lat = {
        "backend": "cellpose", "kind": "domain_specialized",
        "name": "cells · lateral · round_02",
        "path": "artifacts/step2/runs/cells/lateral/round_02/model/"
                "cells_lateral_round_02",
        "run_id": "cells", "round": 2,
        "fingerprint": {"size": 26178012, "mtime_ns": 1791100000000000000},
        "trained_channels": [2, 1],
        "extra": {
            "run_id": "cells", "bucket": "lateral", "round_number": 2,
            "schema_version": "2.1",
            "init": {"mode": "step3_refined", "path": None,
                     "init_combo_text": "From refined bundle mybundle"},
            "inherited_from": {"step3_bundle": "mybundle",
                               "step3_bucket": "lateral"},
            "training": {"n_train_pairs": 37, "n_epochs": 100,
                         "channels_used": ["membrane", "nuclei"]},
            "final_loss": {"train": 0.4213, "val": None},
            "environment": {"cellpose_version": "3.1.1.2",
                            "torch_version": "2.5.1+cu121",
                            "python_version": "3.12.7",
                            "device": "cuda:0",
                            "platform": "Windows-11-10.0.26200-SP0"},
            "model_manifest": "artifacts/step2/runs/cells/lateral/round_02/"
                              "model/manifest.json",
            "zz_foreign_extra": {"deep_key": "deep value"},
        },
        "omnipose_flow_mode": "foreign view key",
    }
    ax = json.loads(json.dumps(lat))
    ax["name"] = "cells · axial · round_02"
    ax["extra"]["bucket"] = "axial"
    return {
        "schema": 1, "recorded_at": NOW.isoformat(),
        "views": {"lateral": lat, "axial": ax,
                  "volumetric": {"kind": "refined", "name": "a 3D model"}},
        "notes": "first note",
        "from_the_future": {"x": 1},
    }


class Saver:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer

    def save(self, notes):
        self.calls.append(notes)
        return self.answer


def make(qapp, block=None, *, locked=False, saver=None, parent=None):
    from qtpy.QtWidgets import QWidget
    from napari_epicycle.fusion_reconstruction._model_provenance_dialog \
        import ModelProvenanceDialog
    host = parent or QWidget()
    saver = saver or Saver()
    dlg = ModelProvenanceDialog(
        host, sample_block() if block is None else block,
        title=PT.PROV_HEADER.format(patch="p01.tif"),
        texts=PT.DIALOG_TEXTS, on_save=saver.save, locked=locked)
    return host, dlg, saver


def _all_text(dlg) -> str:
    from qtpy.QtWidgets import QLabel
    from napari_epicycle.ui_shell._recon_settings_widgets import ZWSP
    return "\n".join(lb.text().replace(ZWSP, "")
                     for lb in dlg.findChildren(QLabel))


def test_every_key_shows_generically(qapp):
    host, dlg, _ = make(qapp)
    try:
        shown = dlg.shown_items()
        keys = {(sec, k) for sec, k, _v in shown}
        for k, _v in P.flatten_items(sample_block()["views"]["lateral"]):
            assert ("lateral", k) in keys, k
        assert ("lateral", "extra.zz_foreign_extra.deep_key") in keys
        assert ("lateral", "omnipose_flow_mode") in keys
        assert ("volumetric", "name") in keys          # a foreign VIEW
        assert ("record", "from_the_future.x") in keys  # a foreign block key
        assert ("record", "schema") in keys
        assert not any(k == "notes" for _s, k, _v in shown)
        text = _all_text(dlg)
        for needle in ("deep value", "foreign view key", "a 3D model",
                       "extra.zz_foreign_extra.deep_key",
                       "cells_lateral_round_02", "Windows-11-10.0.26200-SP0",
                       PT.PROV_VIEW_LATERAL, PT.PROV_VIEW_AXIAL,
                       PT.KIND_WORDS["domain_specialized"],
                       "p01.tif"):
            assert needle in text, needle
    finally:
        host.deleteLater()


def test_a_value_shows_with_the_window_s_empty_word(qapp):
    host, dlg, _ = make(qapp)
    try:
        shown = {(s, k): v for s, k, v in dlg.shown_items()}
        assert shown[("lateral", "extra.final_loss.val")] == PT.PROV_EMPTY
        assert shown[("lateral", "trained_channels")] == "2, 1"
    finally:
        host.deleteLater()


def test_no_block_says_so_and_still_takes_notes(qapp):
    host, dlg, saver = make(qapp, block={})
    try:
        assert PT.PROV_NO_BLOCK in _all_text(dlg)
        dlg.notes_edit.setPlainText("for an old patch")
        dlg.save_btn.click()
        assert saver.calls == ["for an old patch"]
    finally:
        host.deleteLater()


def test_save_hands_the_notes_to_the_caller(qapp):
    host, dlg, saver = make(qapp)
    try:
        assert dlg.notes_edit.toPlainText() == "first note"
        assert not dlg.notes_edit.isReadOnly()
        dlg.notes_edit.setPlainText("second note")
        assert dlg.save_btn.isEnabled()
        dlg.save_btn.click()
        assert saver.calls == ["second note"]
        assert dlg.status_label.text() == PT.PROV_SAVED
        assert not dlg.status_label.isHidden()
    finally:
        host.deleteLater()


@pytest.mark.parametrize("answer,shown", [
    ("A sentence from the caller.", "A sentence from the caller."),
    (False, PT.PROV_SAVE_FAILED),
])
def test_a_refused_save_says_why_on_screen(qapp, answer, shown):
    host, dlg, saver = make(qapp, saver=Saver(answer))
    try:
        dlg.save_btn.click()
        assert dlg.status_label.text() == shown
        assert not dlg.status_label.isHidden()
    finally:
        host.deleteLater()


def test_a_save_that_raises_is_caught(qapp):
    class Boom:
        def save(self, notes):
            raise OSError("disk full")
    host, dlg, _ = make(qapp, saver=Boom())
    try:
        dlg.save_btn.click()
        assert dlg.status_label.text() == PT.PROV_SAVE_FAILED
    finally:
        host.deleteLater()


def test_the_dialog_does_no_file_io():
    src = (_REPO / "napari_epicycle" / "fusion_reconstruction" /
           "_model_provenance_dialog.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = {getattr(n.func, "attr", getattr(n.func, "id", ""))
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for io in ("open", "write_text", "read_text", "write_notes",
               "backfill_manifest", "write_json_atomic"):
        assert io not in calls, io


def test_locked_greys_save_with_the_reason_on_screen(qapp):
    host, dlg, saver = make(qapp, locked=True)
    try:
        host.show()
        dlg.show()
        _pump(qapp)
        assert not dlg.save_btn.isEnabled()
        assert dlg.reason_label.text() == PT.PROV_LOCKED
        assert dlg.reason_label.isVisible()
        dlg.save_btn.click()
        assert saver.calls == []
        dlg.set_locked(False)
        _pump(qapp)
        assert dlg.save_btn.isEnabled() and dlg.reason_label.isHidden()
        dlg.set_locked(True)
        assert not dlg.save_btn.isEnabled() and not dlg.reason_label.isHidden()
    finally:
        dlg.close()
        host.deleteLater()
        _pump(qapp)


def test_unlocked_has_no_reason_line(qapp):
    host, dlg, _ = make(qapp)
    try:
        assert dlg.reason_label.isHidden()
    finally:
        host.deleteLater()


def test_every_widget_is_parented_and_close_deletes(qapp):
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtWidgets import QApplication
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    before = QApplication.topLevelWidgets()      # kept alive: ids stay unique
    host, dlg, _ = make(qapp)
    try:
        new = [w for w in QApplication.topLevelWidgets()
               if not any(w is b for b in before)]
        assert {id(w) for w in new} == {id(host), id(dlg)},             [type(w).__name__ for w in new]
        assert dlg.parent() is host
        assert dlg.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        destroyed = []
        dlg.destroyed.connect(destroyed.append)
        dlg.show()
        _pump(qapp)
        dlg.close_btn.click()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        _pump(qapp)
        assert destroyed
    finally:
        host.deleteLater()


def test_no_lambda_or_partial_is_connected():
    src = (_REPO / "napari_epicycle" / "fusion_reconstruction" /
           "_model_provenance_dialog.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "connect"):
            for arg in node.args:
                assert not isinstance(arg, ast.Lambda), ast.dump(node)
                assert not (isinstance(arg, ast.Call) and getattr(
                    arg.func, "id", getattr(arg.func, "attr", ""))
                    == "partial"), ast.dump(node)
    assert not any(isinstance(n, ast.Lambda) for n in ast.walk(tree))


def test_the_window_words_are_constants():
    for key in ("header", "lateral", "axial", "record", "notes",
                "notes_hint", "save", "close", "locked", "saved",
                "save_failed", "empty", "view_missing", "no_block", "kinds"):
        assert key in PT.DIALOG_TEXTS, key
    for key, value in PT.DIALOG_TEXTS.items():
        if key == "kinds":
            assert value is PT.KIND_WORDS
        else:
            assert any(value is v for k, v in vars(PT).items()
                       if k.isupper() and k != "DIALOG_TEXTS"), key


# ── the size, in the user's font ───────────────────────────────────────────


def measure_all(qapp):
    from qtpy.QtWidgets import QWidget
    from napari_epicycle.fusion_reconstruction._model_provenance_dialog \
        import ModelProvenanceDialog
    out = {"font": qapp.font().family()}
    for name, kw in (("full", {}), ("locked", {"locked": True}),
                     ("no_block", {"block": {}})):
        host = QWidget()
        block = kw.pop("block", sample_block())
        dlg = ModelProvenanceDialog(
            host, block, title=PT.PROV_HEADER.format(patch="p01.tif"),
            texts=PT.DIALOG_TEXTS, on_save=Saver().save, **kw)
        dlg.show()
        for _ in range(10):
            qapp.processEvents()
        out[name] = {"w": dlg.width(), "h": dlg.height(),
                     "min_w": dlg.minimumSizeHint().width(),
                     "hint_w": dlg.sizeHint().width()}
        dlg.close()
        host.deleteLater()
        for _ in range(5):
            qapp.processEvents()
    return out


_SUBPROCESS = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
from tests.fusion_reconstruction import test_model_provenance_dialog as t
print("R6JSON" + json.dumps(t.measure_all(app)), flush=True)
os._exit(0)
"""


@pytest.fixture(scope="module")
def sizes():
    code = f"FONT_DIR = {_FONT_DIR!r}\n" + _SUBPROCESS
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("R6JSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    got = json.loads(lines[-1][len("R6JSON"):])
    print(f"[CEIL] R6 window {json.dumps(got)}")
    return got


def test_measured_in_the_users_font(sizes):
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert sizes["font"] == "Segoe UI"


@pytest.mark.parametrize("state", ["full", "locked", "no_block"])
def test_the_window_fits_the_dock(sizes, state):
    got = sizes[state]
    assert got["w"] <= DOCK_W, got
    assert got["min_w"] <= DOCK_W, got
    assert got["h"] <= 640, got
