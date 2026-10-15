"""D15: the controller identifies the training mode by the substrings
"cumulative" and "scratch", so the Mode combo's two new words keep them.
Every parse site (``_compute_init_hint``, ``_on_training_mode_changed``, the
display refresh, the history-apply loop) must pick the right item."""
import inspect
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QComboBox  # noqa: E402

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (  # noqa: E402
    TrainingAnnotationController as C,
)
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _combo():
    cb = QComboBox()
    cb.addItems([T.MODE_CONTINUE, T.MODE_SCRATCH])
    return cb


def test_each_word_has_exactly_its_own_key_word():
    assert "cumulative" in T.MODE_CONTINUE.lower()
    assert "scratch" not in T.MODE_CONTINUE.lower()
    assert "scratch" in T.MODE_SCRATCH.lower()
    assert "cumulative" not in T.MODE_SCRATCH.lower()


def test_the_real_combo_is_built_with_the_words(qapp):
    from tests.hitl_training_annotation.test_step2_training_hookup import _real_widget
    w = _real_widget(qapp)
    texts = [w.training_mode_combo.itemText(i) for i in range(2)]
    assert texts == [T.MODE_CONTINUE, T.MODE_SCRATCH]


class _Ctrl(C):
    def __init__(self, combo):
        self.ui = type("U", (), {})()
        self.ui.training_mode_combo = combo
        self._init_transparency_loading = False
        self.training_mode_by_bucket = {"lateral": "cumulative", "axial": "cumulative"}
        self.view_target = "lateral"
        self.selected_view = "XY"
        self.current_config = {}
        self._schedule_state_save = lambda: None


def test_the_mode_text_changed_parse(qapp):
    c = _Ctrl(_combo())
    c._refresh_init_hint = lambda: None
    c._on_training_mode_changed(T.MODE_SCRATCH)
    assert c.training_mode == "from_scratch"
    c._on_training_mode_changed(T.MODE_CONTINUE)
    assert c.training_mode == "cumulative"


def test_the_display_refresh_picks_the_right_item(qapp):
    cb = _combo()
    c = _Ctrl(cb)
    name = "_apply_training_mode_for_active_bucket"
    c.training_mode_by_bucket = {"lateral": "from_scratch", "axial": "cumulative"}
    getattr(c, name)()
    assert cb.currentText() == T.MODE_SCRATCH
    c.training_mode_by_bucket = {"lateral": "cumulative", "axial": "cumulative"}
    getattr(c, name)()
    assert cb.currentText() == T.MODE_CONTINUE


def test_the_init_hint_reads_the_mode_by_substring(qapp):
    cb = _combo()
    c = _Ctrl(cb)
    c.working_dir = "/ws"
    c.active_run_id = "Cell"
    c.current_round_by_view = {"lateral": 2, "axial": 0}
    c.ui.model_selector = QComboBox()
    c.ui.model_selector.addItems(["scratch", "cyto3"])
    c._resolve_prior_model_checkpoint = lambda bucket=None: {"spatial": "x"}
    cb.setCurrentIndex(0)
    assert c._compute_init_hint()[1].startswith("Starts from Cell_lateral_round_01")
    cb.setCurrentIndex(1)
    sev, text = c._compute_init_hint()
    assert sev == "apply" and "every round" in text


def test_the_auto_flip_and_history_loops_find_both_items(qapp):
    src = inspect.getsource(C)
    # the auto-flip (``_on_init_widget_changed``) and the history-apply loop
    # match by substring; both words contain their key
    assert 'if "scratch" in mode_combo.itemText(i).lower()' in src
    assert 'if want_scratch and "scratch" in item_text' in src
    assert 'if not want_scratch and "cumulative" in item_text' in src
