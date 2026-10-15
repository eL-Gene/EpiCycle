"""Brick T (module half): the Training builders (offscreen Qt).  Run on its
own with ``-o addopts=""``."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import (QApplication, QComboBox, QLabel, QLineEdit,  # noqa: E402
                            QListWidget, QProgressBar, QPushButton,
                            QRadioButton, QSpinBox, QWidget)

from napari_epicycle.hitl_training_annotation.ui import _step2_training as T  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui.loss_plot_widget import LossPlotWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _setup(qapp):
    host = QWidget()
    mode = QComboBox()
    mode.addItems(["old a", "old b"])
    w = dict(
        model_name_lateral_label=QLabel("-"), model_name_axial_label=QLabel("-"),
        training_mode_combo=mode, model_selector=QComboBox(),
        starting_checkpoint_path=QLineEdit(), starting_checkpoint_browse=QPushButton("Browse"),
        starting_checkpoint_status=QLabel(""), epochs_sb=QSpinBox(), batch_sb=QSpinBox(),
        device_gpu=QRadioButton("GPU"), device_cpu=QRadioButton("CPU"),
    )
    col = T.build_setup_column(parent=host, **w)
    return host, col, w


def _foot(qapp):
    host = QWidget()
    plot = LossPlotWidget()
    tr, ca = QPushButton("x"), QPushButton("y")
    foot = T.build_training_foot(tr, ca, QProgressBar(), QLabel(), plot, parent=host)
    return host, foot, plot


class TestSentences:
    def test_mode_key_words(self, qapp):
        _h, col, w = _setup(qapp)
        texts = [w["training_mode_combo"].itemText(i) for i in range(2)]
        assert "cumulative" in texts[0].lower() and "scratch" not in texts[0].lower()
        assert "scratch" in texts[1].lower() and "cumulative" not in texts[1].lower()
        assert texts == [T.MODE_CONTINUE, T.MODE_SCRATCH]

    def test_six_hint_cases(self):
        kw = dict(source="cyto3", bucket="lateral", display="cyto3",
                  model="Cell_lateral_round_02", n=3, base="cyto3", file="a.model")
        assert len(T.HINT_CASES) == 6
        outs = {c: T.hint_text(c, **kw) for c in T.HINT_CASES}
        assert len(set(outs.values())) == 6
        assert "Cell_lateral_round_02" in outs["cumulative_prior"]
        assert "a.model" in outs["scratch_file"] and "every round" in outs["scratch_base"]
        with pytest.raises(KeyError):
            T.hint_text("nope")

    def test_stage_line_cases(self):
        run = T.compose_training_stage_line(run="Cell", bucket="lateral", round_number=4,
                                            epoch=412, epochs=1000, loss=0.8123)
        assert run == "Cell · lateral round 4 · epoch 412 / 1000 · loss 0.81"
        idle = T.compose_training_stage_line(run="Cell", bucket="lateral",
                                             file_name="Cell_lateral_round_04", n_pairs=13)
        assert idle.startswith(
            "Ready · Train writes Cell_lateral_round_04 from 13 lateral pairs")
        assert T.compose_training_stage_line() == T.STAGE_IDLE_BARE

    def test_counts(self):
        assert "(8 from rounds 0–2)" in T.staged_count_text(5, 13, 3, "lateral")
        assert "from rounds" not in T.staged_count_text(5, 5, 0, "lateral")
        assert "nothing finalized" in T.staged_other_text("axial", 0)
        assert "3 finalized" in T.staged_other_text("axial", 3)
        assert T.writes_text("Cell_lateral_round_03", 3) == "Cell_lateral_round_03 · round 3"


class TestSetup:
    def test_grey_only_on_override(self, qapp):
        _h, col, w = _setup(qapp)
        for sev, grey in (("apply", False), ("warn", False), ("neutral", False),
                          ("override", True)):
            col.set_hint(sev, "t")
            for k in ("model_selector", "starting_checkpoint_path",
                      "starting_checkpoint_browse"):
                assert w[k].isEnabled() is (not grey), (sev, k)
            assert col.grey_reason_label.isHidden() is (not grey)
        col.set_hint("apply", "t")
        assert col.model_selector.isEnabled()

    def test_override_restores_prior_state(self, qapp):
        _h, col, w = _setup(qapp)
        w["model_selector"].setEnabled(False)
        col.set_hint("override", "t")
        col.set_hint("apply", "t")
        assert not w["model_selector"].isEnabled()
        assert w["starting_checkpoint_path"].isEnabled()

    def test_warn_is_the_only_amber_block(self, qapp):
        _h, col, _w = _setup(qapp)
        col.set_hint("warn", "x")
        assert "rgba(212, 168, 67" in col.hint_label.styleSheet()
        col.set_hint("apply", "x")
        assert "rgba(212, 168, 67" not in col.hint_label.styleSheet()


class TestFoot:
    def test_two_distinct_buttons_one_primary(self, qapp):
        _h, foot, _p = _foot(qapp)
        assert foot.train_btn is not foot.train_cancel_btn
        assert foot.train_btn.text() == "Train" and foot.train_cancel_btn.text() == "Cancel"
        assert foot.train_btn.styleSheet() == T.PRIMARY_BTN_STYLE
        assert foot.train_cancel_btn.styleSheet() != foot.train_btn.styleSheet()
        assert not foot.train_cancel_btn.isEnabled()
        foot.set_running(True)
        assert not foot.train_btn.isEnabled() and foot.train_cancel_btn.isEnabled()

    def test_loss_opens_and_closes(self, qapp):
        _h, foot, plot = _foot(qapp)
        assert not foot.loss.is_open() and foot.loss.body.isHidden()
        assert plot.grade_by_round is True
        foot.open_for_round("lat r4 · epoch 1 · 0.9")
        assert foot.loss.is_open() and not foot.loss.body.isHidden()
        assert foot.loss.caption_label.text().startswith("lat r4")
        foot.close_idle()
        assert not foot.loss.is_open() and foot.loss.body.isHidden()
        assert foot.loss.caption_label.text() == plot.grading_caption()


class TestPanel:
    def _panel(self, qapp):
        host = QWidget()
        _h, col, _w = _setup(qapp)
        staged = T.build_staged_column(QListWidget(), QLabel(), parent=host)
        _h2, foot, plot = _foot(qapp)
        panel = T.build_training_panel(col, staged, foot, parent=host)
        return host, panel, plot

    def test_splitter_minimums(self, qapp):
        from qtpy.QtCore import Qt
        _h, p, _pl = self._panel(qapp)
        assert p.splitter.orientation() == Qt.Vertical
        assert p.splitter.count() == 2 and not p.splitter.childrenCollapsible()
        assert p.columns_splitter.minimumHeight() == T.SPLIT_TOP_MIN_H
        assert p.columns_splitter.widget(0).minimumWidth() == T.LEFT_MIN_W
        assert p.columns_splitter.widget(1).minimumWidth() == T.RIGHT_MIN_W
        assert p.foot.loss._plot.minimumHeight() >= T.LOSS_MIN_H

    def test_lock_line(self, qapp):
        _h, p, _pl = self._panel(qapp)
        assert p.lock_line.isHidden()
        p.on_training_locked(True, "Cell_lateral_round_04", "lateral")
        assert p.lock_line.text() == (
            "Round 4 (lateral) is training — the staged list "
            "and the setup are read-only until it finishes.")
        assert not p.lock_line.isHidden()
        p.on_training_locked(False, "Cell_lateral_round_04", "lateral")
        assert p.lock_line.isHidden() and not p.foot.train_cancel_btn.isEnabled()

    def test_train_started_hook(self, qapp):
        _h, p, plot = self._panel(qapp)
        p.on_train_started("axial", "Cell_axial_round_02")
        assert p.foot.loss.is_open() and plot._live_family == "axial"
        assert not p.foot.train_btn.isEnabled() and p.foot.train_cancel_btn.isEnabled()

    def test_every_widget_has_a_parent(self, qapp):
        host, p, _pl = self._panel(qapp)
        assert not [w for w in p.findChildren(QWidget) if w.parent() is None]
        for w in (p.foot.train_btn, p.foot.train_cancel_btn, p.foot.stage_label,
                  p.foot.loss, p.lock_line, p.splitter, p.setup_column, p.staged_column):
            assert w.parent() is not None
        assert p.parent() is host
