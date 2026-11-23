"""The blank "Channel to segment" and the one-channel patches, at the foot.

* While "Channel to segment" is blank, Predict Selected is grey and the
  foot says "Pick the channel to segment." -- unless every runnable ticked
  patch has one channel (Cellpose reads such a patch as gray whatever the
  box says).
* A ticked one-channel patch is named on the ready line ("{patch}: one
  channel, runs as 0: gray.", or the plural) -- but never after "Predict
  the ticked patches first.", one sentence per state -- and ticking it
  never touches the boxes, Start from or the run's memory.
* One rule decides "one channel" (no C axis, or C of length 1) for the
  foot and for Predict.
* A blank box never reaches a record as a number.

A real panel + controller; the ticks, the runnable list and the cached
axes / shapes are given by hand (the folder load is tested elsewhere).
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _step3_pr_text as P  # noqa: E402
from napari_epicycle.fusion_reconstruction._step3_channels import (  # noqa: E402
    is_one_channel, one_channel_names,
)

from ._one_channel import (  # noqa: E402
    ZCYX_C1_SHAPE, ZYX_SHAPE, write_two_channel_patch, write_zyx_patch,
)

ONE = "Wound_patch_5.tiff"
ONE_C1 = "Wound_patch_6.tiff"
TWO = "Wound_patch_2.tiff"

AXES = {ONE: "ZYX", ONE_C1: "ZCYX", TWO: "CZYX"}
SHAPES = {ONE: list(ZYX_SHAPE), ONE_C1: list(ZCYX_C1_SHAPE),
          TWO: [2] + list(ZYX_SHAPE)}


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def w(app, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    widget = FusionReconstructionCuration(viewer=None)
    yield widget
    timer = getattr(widget.controller.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    widget.deleteLater()
    app.processEvents()


def _open(w, ws: Path, start_from="epicycle"):
    from napari_epicycle.fusion_reconstruction import _paths
    (ws / _paths.STEP3_STATE_FILENAME).write_text(json.dumps({
        "schema_version": 3, "active_run_id": "default",
        "runs": {"default": {"patches": {}, "start_from": start_from}}}),
        encoding="utf-8")
    c = w.controller
    c.working_dir = str(ws)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(ws))
    c._restore_start_from()
    w.lateral_model_path.setText("nuclei")
    w.axial_model_path.setText("nuclei")
    return c


def _tick(c, monkeypatch, names):
    names = list(names)
    monkeypatch.setattr(c, "_ticked_patch_names", lambda: list(names))
    monkeypatch.setattr(c, "_pr_runnable",
                        lambda ns: (list(ns), [], []))
    monkeypatch.setattr(c, "_session_axes",
                        lambda ns: {n: AXES[n] for n in ns if n in AXES})
    monkeypatch.setattr(c, "_session_shapes",
                        lambda ns: {n: SHAPES[n] for n in ns if n in SHAPES})


def _foot(w, c):
    seen = []
    orig = w.show_pr_status

    def spy(text, *a, **k):
        seen.append(text)
        return orig(text, *a, **k)
    w.show_pr_status = spy
    try:
        c._refresh_pr_foot_gates(keep_status=False)
    finally:
        w.show_pr_status = orig
    return seen[-1] if seen else None


# ------------------------------------------------------------- the one rule

@pytest.mark.parametrize("axes,shape,one", [
    ("ZYX", [4, 8, 8], True),
    ("ZCYX", [4, 1, 8, 8], True),
    ("CZYX", [1, 4, 8, 8], True),
    ("ZCYX", [4, 2, 8, 8], False),
    ("CZYX", [2, 4, 8, 8], False),
    ("ZCYX", None, False),          # a C axis whose length is not known
    ("CZYX", [8, 8], True),         # a 2D image
    (None, [8, 8], True),
    ("", [4, 8, 8], False),         # axes not known
    (None, None, False),
])
def test_one_rule_for_one_channel(axes, shape, one):
    assert is_one_channel(axes, shape) is one


def test_the_fixture_patches_read_as_one_channel(tmp_path):
    import tifffile

    from napari_epicycle.fusion_reconstruction.qc_features import (
        resolve_patch_axes,
    )
    for path in (write_zyx_patch(tmp_path, "a.tif"),
                 write_zyx_patch(tmp_path, "b.tif", c1=True)):
        image = tifffile.imread(str(path))
        axes = resolve_patch_axes(image, image_path=str(path),
                                  fallback="CZYX")
        assert is_one_channel(axes, image.shape), (path.name, axes,
                                                   image.shape)
    two = write_two_channel_patch(tmp_path)
    image = tifffile.imread(str(two))
    axes = resolve_patch_axes(image, image_path=str(two), fallback="CZYX")
    assert not is_one_channel(axes, image.shape)


def test_one_channel_names_keeps_the_order_and_skips_the_unknown():
    names = [TWO, ONE, "unknown.tif", ONE_C1]
    assert one_channel_names(names, AXES, SHAPES) == [ONE, ONE_C1]


def test_the_sentence_is_step_4s():
    from napari_epicycle.segmentation import _step4_text as S4T
    assert P.S3_GATE_PICK_CHANNEL == S4T.SEG_PICK_CHANNEL


# ------------------------------------------------------------- the gate (I5)

def test_a_blank_box_greys_predict_with_a_two_channel_patch(w, tmp_path,
                                                            monkeypatch):
    c = _open(w, tmp_path)
    _tick(c, monkeypatch, [TWO])
    w.chan1_combo.setCurrentIndex(-1)
    assert _foot(w, c) == P.S3_GATE_PICK_CHANNEL
    assert not w.predict_selected_btn.isEnabled()
    w.chan1_combo.setCurrentIndex(1)
    assert _foot(w, c).startswith(P.S3_GATE_READY.format(n=1))
    assert w.predict_selected_btn.isEnabled()


def test_the_gate_order_model_first_then_the_channel(w, tmp_path,
                                                     monkeypatch):
    """With nothing ticked the tick sentence wins: the pick is asked for
    only once a patch that needs it is ticked."""
    c = _open(w, tmp_path)
    _tick(c, monkeypatch, [])
    w.chan1_combo.setCurrentIndex(-1)
    assert _foot(w, c) == P.S3_GATE_NOTHING_TICKED
    _tick(c, monkeypatch, [TWO])
    assert _foot(w, c) == P.S3_GATE_PICK_CHANNEL
    w.lateral_model_path.setText("")
    assert _foot(w, c) == P.S3_GATE_PICK_MODEL


def test_the_rule_is_the_rows_channel_count():
    from napari_epicycle.fusion_reconstruction._step3_rows import (
        channel_count,
    )
    for name in AXES:
        assert is_one_channel(AXES[name], SHAPES[name]) is (
            channel_count(AXES[name], SHAPES[name]) == 1)


@pytest.mark.parametrize("ticked", [[ONE], [ONE_C1], [ONE, ONE_C1]])
def test_every_ticked_patch_one_channel_lets_a_blank_box_predict(
        w, tmp_path, monkeypatch, ticked):
    c = _open(w, tmp_path)
    _tick(c, monkeypatch, ticked)
    w.chan1_combo.setCurrentIndex(-1)
    text = _foot(w, c)
    assert w.predict_selected_btn.isEnabled()
    assert text.startswith(P.S3_GATE_READY.format(n=len(ticked)))
    _tick(c, monkeypatch, ticked + [TWO])
    assert _foot(w, c) == P.S3_GATE_PICK_CHANNEL
    assert not w.predict_selected_btn.isEnabled()


# ------------------------------------------------- the ready line's clause

def _predicted(c, *names):
    for n in names:
        c.state_manager.set_patch_status(n, "predicted")


def test_the_ready_line_names_one_one_channel_patch(w, tmp_path,
                                                    monkeypatch):
    c = _open(w, tmp_path)
    w.chan1_combo.setCurrentIndex(1)
    _tick(c, monkeypatch, [ONE])
    _predicted(c, ONE)
    monkeypatch.setattr(w, "pr_status_fits", lambda text: True)
    join = P.S3_GATE_JOIN.join
    assert _foot(w, c) == join([
        P.S3_GATE_READY.format(n=1),
        P.S3_GATE_ONE_CHANNEL.format(patch=ONE)])


def test_one_sentence_per_state_never_after_predict_first(w, tmp_path,
                                                          monkeypatch):
    """Nothing ticked is predicted yet: the line says "Predict the ticked
    patches first." and stops there -- the one-channel sentence is not
    added after it (three lines in the 578 px dock's two-line foot)."""
    c = _open(w, tmp_path)
    w.chan1_combo.setCurrentIndex(1)
    _tick(c, monkeypatch, [ONE])
    monkeypatch.setattr(w, "pr_status_fits", lambda text: True)
    assert _foot(w, c) == P.S3_GATE_JOIN.join([
        P.S3_GATE_READY.format(n=1), P.S3_GATE_NOTHING_TO_RECONSTRUCT])
    _tick(c, monkeypatch, [ONE, ONE_C1])
    assert P.S3_GATE_ONE_CHANNEL_PLURAL.format(k=2) not in _foot(w, c)


def test_the_ready_line_counts_several(w, tmp_path, monkeypatch):
    c = _open(w, tmp_path)
    w.chan1_combo.setCurrentIndex(1)
    _tick(c, monkeypatch, [ONE, ONE_C1, TWO])
    _predicted(c, TWO)
    monkeypatch.setattr(w, "pr_status_fits", lambda text: True)
    assert _foot(w, c).endswith(P.S3_GATE_ONE_CHANNEL_PLURAL.format(k=2))


def test_a_narrow_foot_keeps_the_one_channel_sentence(w, tmp_path,
                                                     monkeypatch):
    """The sentence has no shorter form: a foot too narrow for any try
    still names the patch."""
    c = _open(w, tmp_path)
    w.chan1_combo.setCurrentIndex(1)
    _tick(c, monkeypatch, [ONE, TWO])
    _predicted(c, ONE, TWO)
    monkeypatch.setattr(w, "pr_status_fits", lambda text: False)
    assert P.S3_GATE_ONE_CHANNEL.format(patch=ONE) in _foot(w, c)
    assert not hasattr(P, "S3_GATE_ONE_CHANNEL_SHORT")


def test_the_one_channel_sentence_leaves_the_protected_clause_out(
        w, tmp_path, monkeypatch):
    """One sentence per state: with a ticked protected patch as well, the
    line is the one-channel sentence alone, so it stays whole."""
    c = _open(w, tmp_path)
    w.chan1_combo.setCurrentIndex(1)
    _tick(c, monkeypatch, [ONE])
    _predicted(c, ONE)
    c.state_manager.set_patch_status("done.tif", "finalized")
    monkeypatch.setattr(c, "_ticked_patch_names",
                        lambda: [ONE, "done.tif"])
    monkeypatch.setattr(c, "_pr_runnable",
                        lambda ns: ([n for n in ns if n != "done.tif"],
                                    ["done.tif"], []))
    monkeypatch.setattr(w, "pr_status_fits", lambda text: True)
    assert _foot(w, c) == P.S3_GATE_JOIN.join([
        P.S3_GATE_READY.format(n=2),
        P.S3_GATE_ONE_CHANNEL.format(patch=ONE)])


def _words_wide(w, factor):
    """Words until the text is *factor* times the status line's width."""
    from napari_epicycle.fusion_reconstruction import ui as UI
    lbl = w.pipeline_status_label
    lbl.setStyleSheet(UI._PR_STATUS_STYLES["neutral"])
    lbl.ensurePolished()
    room = max(lbl.contentsRect().width(), 200)
    fm = lbl.fontMetrics()
    words = []
    while fm.horizontalAdvance(" ".join(words)) < factor * room:
        words.append(f"word{len(words)}")
    return " ".join(words)


def test_the_fit_rule_is_two_lines(w):
    """``pr_status_fits`` says whether the foot's TWO lines hold a text
    whole (as the guide says): one line fits, a text that wraps once
    fits, one that would need a third line does not."""
    assert w.pr_status_fits("Ready · 1 ticked")
    assert w.pr_status_fits(_words_wide(w, 1.4))
    assert not w.pr_status_fits(_words_wide(w, 3.0))


def test_the_one_channel_sentences_end_with_a_full_stop():
    for t in (P.S3_GATE_ONE_CHANNEL.format(patch=ONE),
              P.S3_GATE_ONE_CHANNEL_PLURAL.format(k=2)):
        assert t.endswith("."), t


def test_the_example_line(w, tmp_path, monkeypatch):
    """The guide's example, word for word."""
    assert (P.S3_GATE_JOIN.join([P.S3_GATE_READY.format(n=1),
                                 P.S3_GATE_ONE_CHANNEL.format(patch=ONE)])
            == "Ready · 1 ticked · Wound_patch_5.tiff: one channel, "
               "runs as 0: gray.")


# ----------------------------------- ticking never touches the setting (I6 b)

@pytest.mark.parametrize("start_from", ["epicycle", "custom"])
def test_ticking_a_one_channel_patch_writes_no_widget(w, tmp_path,
                                                      monkeypatch,
                                                      start_from):
    c = _open(w, tmp_path, start_from=start_from)
    w.chan1_combo.setCurrentIndex(1)
    w.chan2_combo.setCurrentIndex(0)
    if start_from == "epicycle":
        c._apply_start_from("epicycle", save=False)
    before = (c.state_manager.get_run_channels(), c._start_from_key(),
              [w.chan1_combo.itemText(i) for i in range(w.chan1_combo.count())],
              w.chan1_combo.isEnabled(), w.chan2_combo.isEnabled())
    fired = []
    w.chan1_combo.currentIndexChanged.connect(fired.append)
    w.chan2_combo.currentIndexChanged.connect(fired.append)
    try:
        for ticked in ([ONE], [ONE, TWO], [TWO], [], [ONE_C1]):
            _tick(c, monkeypatch, ticked)
            _foot(w, c)
    finally:
        w.chan1_combo.currentIndexChanged.disconnect(fired.append)
        w.chan2_combo.currentIndexChanged.disconnect(fired.append)
    assert fired == []
    assert (w.chan1_combo.currentIndex(), w.chan2_combo.currentIndex()) == \
        (1, 0)
    assert (c.state_manager.get_run_channels(), c._start_from_key(),
            [w.chan1_combo.itemText(i) for i in range(w.chan1_combo.count())],
            w.chan1_combo.isEnabled(), w.chan2_combo.isEnabled()) == before


# ------------------------------------------ the blank never becomes a number

def test_qcs_fallback_reads_a_blank_box_as_gray(w, tmp_path, monkeypatch):
    c = _open(w, tmp_path)
    monkeypatch.setattr(c, "_resolve_patch_payload", lambda name: None)
    logs = []
    c._append_log = logs.append
    w.chan1_combo.setCurrentIndex(-1)
    assert c._get_active_seg_channel() == 0
    assert P.S3_LOG_QC_BLANK_CHANNEL in logs
    logs.clear()
    w.chan1_combo.setCurrentIndex(2)
    assert c._get_active_seg_channel() == 2
    assert logs == []


def test_the_reconstruct_fallback_records_a_blank_box_as_none(w, tmp_path):
    c = _open(w, tmp_path)
    w.chan1_combo.setCurrentIndex(-1)
    seg = c._collect_seg_config()
    assert seg.channels[0] == -1
    recorded, extra = c._recon_recorded_seg("P1.tif", seg, 0.0)
    assert recorded.channels is None
    assert extra["seg_config_source"] == "settings"
    w.chan1_combo.setCurrentIndex(2)
    recorded, _ = c._recon_recorded_seg("P1.tif", c._collect_seg_config(),
                                        0.0)
    assert list(recorded.channels) == [2, 1]


def test_a_stray_predict_with_a_blank_box_refuses_a_two_channel_patch(
        w, tmp_path, monkeypatch):
    c = _open(w, tmp_path)
    c.loaded_stack_dir = str(tmp_path)
    _tick(c, monkeypatch, [TWO])
    monkeypatch.setattr(c, "_pr_check_files", lambda names: None)
    w.chan1_combo.setCurrentIndex(-1)
    logs = []
    c._append_log = logs.append
    c._predict_all()
    assert P.S3_GATE_PICK_CHANNEL in logs
    assert not getattr(c, "_prediction_running", False)
