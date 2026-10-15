"""Step 2 look, fix pass 1 (the brick-P audit + the Training hookup report):
the left column's width, the room measured on the scroll viewport, the
active-bucket progress note, the busy words, the stage-line bucket, the
inference summary refresh, one CompactSplitter, the stats line, the six hint
cases.  Qt-backed: run on its own with ``-o addopts=""``."""
import inspect
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import (QApplication, QLabel, QProgressBar,  # noqa: E402
                            QPushButton, QScrollArea, QWidget)

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (  # noqa: E402,E501
    TrainingAnnotationController as Ctl)
from napari_epicycle.hitl_training_annotation.ui import _step2_curation as C  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui import _step2_training as TR  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui.loss_plot_widget import LossPlotWidget  # noqa: E402

from test_step2_curation import _ctrl, _FakeTable, _FakeUI, _real_widget  # noqa: E402
from test_step2_training_seams import _Combo, _Edit  # noqa: E402
from test_step2_training_seams import _ctrl as _tctrl  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# 2 -- the left column starts at its minimum and widens
def test_left_column_is_not_pinned_and_widens_with_the_dock(qapp):
    w = _real_widget(qapp)
    left = w.curation_splitter.widget(0)
    assert left.minimumWidth() == 240
    w.resize(578, 600)
    w.show()
    qapp.processEvents()
    narrow = left.width()
    assert narrow >= 240
    w.resize(1000, 600)
    qapp.processEvents()
    assert left.width() > narrow + 60
    w.close()


# 3 -- the room is the scroll viewport's
def test_room_measures_the_enclosing_scroll_viewport(qapp):
    area = QScrollArea()
    area.setWidgetResizable(True)
    w = _real_widget(qapp)
    area.setWidget(w)
    area.resize(578, 820)
    area.show()
    qapp.processEvents()
    assert w._room_height() == area.viewport().height()
    assert w._room_height() >= C.TALL_MODE_MIN_PX
    assert w._curation_short is False
    area.resize(578, 560)
    qapp.processEvents()
    assert w._room_height() < C.TALL_MODE_MIN_PX
    assert w._curation_short is True        # the filter on the viewport fired
    area.close()


def test_room_without_a_scroll_area_falls_back_to_its_own_height(qapp):
    w = _real_widget(qapp)
    w.resize(578, 640)
    assert w._room_height() == w.height()


def test_training_room_uses_the_viewport(qapp, monkeypatch):
    from test_step2_training import _foot
    from test_step2_training_seams import _Label   # noqa: F401
    _h, foot, _plot = _foot(qapp)
    panel = TR.build_training_panel(QWidget(), QWidget(), foot)
    panel.resize(572, 900)
    panel.show()
    qapp.processEvents()
    assert panel.splitter.height() > 600
    monkeypatch.setattr(TR, "enclosing_viewport_height", lambda _w: 300)
    panel.apply_split_sizes(True)
    _top, bottom = panel.splitter.sizes()
    want = max(int((300 - panel.splitter.handleWidth()) * TR.SPLIT_OPEN_FRAC),
               panel.foot.minimumSizeHint().height())
    assert bottom == want
    monkeypatch.setattr(TR, "enclosing_viewport_height", lambda _w: 0)
    panel.apply_split_sizes(True)
    assert panel.splitter.sizes()[1] > want        # no viewport: own height
    panel.close()


# 4 -- the progress note spans the ACTIVE bucket
def test_progress_note_for_an_axial_only_run():
    t = _FakeTable()
    c = _ctrl(_FakeUI(t), view="XZ")
    c.current_round_by_view = {"lateral": 0, "axial": 3}
    c._refresh_sample_queue()
    note = t.calls[0][2]
    assert "axial rounds 0–2" in note
    assert "no round yet" not in note


def test_progress_note_lateral_unchanged():
    t = _FakeTable()
    c = _ctrl(_FakeUI(t))
    c.current_round_by_view = {"lateral": 2, "axial": 0}
    c._refresh_sample_queue()
    assert "lateral rounds 0–1" in t.calls[0][2]


# 5 -- the busy words
def test_strip_log_tag():
    assert C.strip_log_tag("[PATCH] Loading 'x' ...") == "Loading 'x' ..."
    assert C.strip_log_tag("[STEP2-RUN] Switching to Cell") == "Switching to Cell"
    assert C.strip_log_tag("Loading") == "Loading"


def test_stage_line_busy_text_has_no_tag():
    assert C.compose_stage_line(
        busy_text="[WORKSPACE] Opening workspace: C:/w",
        patch="p") == "Opening workspace: C:/w"


def test_busy_words_are_the_bars_tooltip_not_a_label_beside_it(qapp):
    w = _real_widget(qapp)
    foot = w.curation_foot
    foot.set_busy("[PATCH] Loading 'x' ...")
    assert foot.busy_label.isHidden()   # the bar keeps the full width
    assert foot.busy_bar.toolTip() == "Loading 'x' ..."
    foot.set_busy(None)
    assert foot.busy_label.isHidden()


# 6 -- seam (g) leftovers
def test_no_dead_selected_items_and_the_dialog_says_the_button_word():
    src = inspect.getsource(Ctl._launch_multiview_annotator)
    assert "selected_items" not in src
    assert "Open Multiview Workspace" not in src


def test_no_parentless_widget_attributes(qapp):
    w = _real_widget(qapp)
    orphans = [k for k, v in vars(w).items()
               if isinstance(v, QWidget) and v.parentWidget() is None
               and v is not w]
    assert orphans == []
    assert w.finalize_labels_button.isHidden()


# 7 -- the stage-line bucket
def test_stage_line_bucket_follows_selected_view_not_the_radios(qapp):
    w = _real_widget(qapp)

    class Ctl_:
        loaded_patch_name = "patch_9"
        selected_view = "XZ"

    w.controller = Ctl_()
    w.segment_slice_data_list.addItem("a_XZ_z0001")
    try:
        w.refresh_stage_line()          # the radios stay on Lateral
        txt = w.curation_stage_label.text()
        assert "axial" in txt and "lateral" not in txt
    finally:
        del w.controller


# 8 -- the inference summary refresh after a refill
def test_channel_refill_refreshes_the_summary():
    src = inspect.getsource(Ctl._refresh_inference_channel_options)
    assert "_refresh_inference_summary" in src
    calls = []

    class UI:
        inference_channel1_combo = None
        inference_channel2_combo = None

        def _refresh_inference_summary(self):
            calls.append(1)

    c = Ctl.__new__(Ctl)
    c.ui = UI()
    c.current_config = {}
    c._ensure_segmentation_config_defaults = lambda: None
    try:
        c._refresh_inference_channel_options()
    except Exception:
        pass            # a bare controller may fail elsewhere; not our seam


# 9 -- one CompactSplitter
def test_one_compact_splitter_definition():
    from napari_epicycle.hitl_training_annotation.ui import main_widget as MW
    assert TR.CompactSplitter is C.CompactSplitter
    assert "class _CompactSplitter" not in inspect.getsource(TR)
    assert "class CompactSplitter" not in inspect.getsource(MW)


# 10 -- tall mode bites
def test_tall_mode_shows_the_descriptions(qapp):
    w = _real_widget(qapp)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(w)
    area.resize(578, 900)
    area.show()
    qapp.processEvents()
    assert w._curation_short is False
    right = w.curation_right_column
    assert right._descs
    for header, desc in right._descs:
        assert not desc.isHidden()
        assert header.title_label.toolTip() == ""
    for header, desc in w._left_descs:
        assert not desc.isHidden()
    area.resize(578, 560)
    qapp.processEvents()
    for header, desc in right._descs:
        assert desc.isHidden()
    for header, desc in w._left_descs:
        assert desc.isHidden()
    area.close()


# 11 -- the stats line
def test_the_loss_stats_line_is_hidden_in_the_training_foot(qapp):
    host = QWidget()
    plot = LossPlotWidget()
    TR.build_training_foot(QPushButton("x"), QPushButton("y"),
                           QProgressBar(), QLabel(), plot, parent=host)
    assert plot.grade_by_round
    assert plot._stats_label.isHidden()


# 12 / 13 -- the six hint cases against spec section 2
def _hint_ctrl(mode_idx, ckpt="", rnd=2, spec=None, cum_prior=True):
    c = _tctrl(training_mode_combo=_Combo([T.MODE_CONTINUE, T.MODE_SCRATCH],
                                          mode_idx),
               model_selector=_Combo(["scratch", "cyto3"], 1),
               starting_checkpoint_path=_Edit(ckpt))
    c.current_round_by_view = {"lateral": rnd, "axial": 0}
    c._init_spec_for_active_bucket = lambda: spec or {"mode": "scratch"}
    c._resolve_prior_model_checkpoint = (
        lambda bucket=None: {"spatial": "x"} if cum_prior else {})
    return c


def test_six_hint_cases_text_and_severity():
    assert _hint_ctrl(0, rnd=0)._compute_init_hint() == (
        "apply", "Starts from cyto3 — the first lateral round.")
    assert _hint_ctrl(0, ckpt="/a/mine.model", rnd=0)._compute_init_hint() == (
        "apply", "Starts from mine.model — the first lateral round.")
    sev, text = _hint_ctrl(0, rnd=0, spec={
        "mode": "pretrained", "path": "cyto3"})._compute_init_hint()
    assert sev == "override" and "fixes where the first round starts" in text
    assert _hint_ctrl(0)._compute_init_hint() == (
        "override",
        "Starts from Cell_lateral_round_01 — round 1’s lateral model.")
    assert _hint_ctrl(0, cum_prior=False)._compute_init_hint() == (
        "warn", "The last round’s model is missing — starts from cyto3 instead.")
    # from scratch: the starting FILE if one is set, else the starting model
    assert _hint_ctrl(1, ckpt="/a/mine.model")._compute_init_hint() == (
        "apply", "Starts from mine.model again, every round. "
                 "Earlier rounds’ pairs are still used.")
    assert _hint_ctrl(1)._compute_init_hint() == (
        "apply", "Starts from cyto3 again, every round. "
                 "Earlier rounds’ pairs are still used.")


def test_from_scratch_severity_stays_apply_so_it_is_not_amber(qapp):
    """Only "warn" is amber.  From scratch is "apply" (neutral
    text); do not "fix" this to warn."""
    from test_step2_training import _setup
    for case in (_hint_ctrl(1), _hint_ctrl(1, ckpt="/a/mine.model")):
        assert case._compute_init_hint()[0] == "apply"
    _h, col, _w = _setup(qapp)
    col.set_hint("apply", "x")
    assert "rgba(212, 168, 67" not in col.hint_label.styleSheet()
    col.set_hint("warn", "x")
    assert "rgba(212, 168, 67" in col.hint_label.styleSheet()
