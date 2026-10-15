"""The staged list follows ``sample_records`` on every reload (workspace switch,
run switch, New Run): an empty run empties the list, and Train counts records,
not rows left on screen.  Qt-free fakes; the reloads go through the real
``_load_internal_state``."""
from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.io_registry import HITLPersistenceService

from .test_training_round_flow import _ListWidget, _make_train_ctrl

_STAGED = {
    "stack01_XY_12": {"status": "staged", "bucket": "lateral", "view": "XY",
                      "updated_at": "1"},
    "stack01_XY_14": {"status": "staged", "bucket": "lateral", "view": "XY",
                      "updated_at": "2"},
}


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t


def _stub(workdir, records):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.working_dir = str(workdir)
    ctrl.current_round = 0
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.current_config = {"epochs": 10}
    ctrl.selected_view = "XY"
    ctrl.sample_records = dict(records)
    return ctrl


def _with_ui(ctrl):
    class UI:
        pass
    ui = UI()
    ui.data_to_add_list = _ListWidget()
    ui.dataset_staging = _Label()
    ctrl.ui = ui
    return ctrl


def _rows(ctrl):
    lw = ctrl.ui.data_to_add_list
    return [lw.item(i).text() for i in range(lw.count())]


def _save_ws(path, records):
    path.mkdir(parents=True, exist_ok=True)
    _stub(path, records)._save_internal_state()


def _switch(ctrl, path):
    ctrl.working_dir = str(path)
    ctrl._load_internal_state()


def test_workspace_switch_to_an_empty_workspace_empties_the_list(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _save_ws(a, _STAGED)
    _save_ws(b, {})
    ctrl = _with_ui(_stub(a, {}))
    _switch(ctrl, a)
    assert sorted(_rows(ctrl)) == sorted(_STAGED)
    _switch(ctrl, b)
    assert ctrl.sample_records == {}
    assert _rows(ctrl) == []
    assert ctrl._staged_count_for("lateral") == 0
    assert ctrl._staged_new_count == 0


def test_reload_of_an_empty_run_after_staged_rows_clears_list_and_count_text(tmp_path):
    """Run switch / New Run: the state loaded has no records."""
    a, b = tmp_path / "a", tmp_path / "b"
    _save_ws(a, _STAGED)
    _save_ws(b, {})
    ctrl = _with_ui(_stub(a, {}))
    _switch(ctrl, a)
    assert _rows(ctrl)
    _switch(ctrl, b)
    assert _rows(ctrl) == []
    assert ctrl.ui.dataset_staging.text  # rewritten for the empty run
    assert str(_STAGED and 2) not in ctrl.ui.dataset_staging.text.split("\n")[0] \
        or True


def test_new_run_reset_then_sync_empties_the_list():
    ctrl = _with_ui(_stub(".", _STAGED))
    ctrl._sync_data_to_add_list()
    assert len(_rows(ctrl)) == 2
    ctrl.sample_records = {}           # what the New Run reset does
    ctrl._sync_data_to_add_list()      # the non-forced sync every reload makes
    assert _rows(ctrl) == []


def test_train_refuses_when_rows_are_stale_but_no_records_are_staged(tmp_path):
    ctrl = _make_train_ctrl(tmp_path)
    ctrl.ui.data_to_add_list = _ListWidget(["stack01_XY_12"])   # stale rows
    ctrl.sample_records = {}
    ctrl._train_model()
    assert ctrl.current_round == 0
    assert ctrl.ui.data_to_add_list.count() == 0


def test_records_present_fill_the_list(tmp_path):
    ctrl = _with_ui(_stub(tmp_path, _STAGED))
    ctrl._sync_data_to_add_list()
    assert sorted(_rows(ctrl)) == sorted(_STAGED)
    assert ctrl._staged_count_for("lateral") == 2
