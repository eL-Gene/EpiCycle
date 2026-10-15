"""Model Refinement look, Brick B2 -- the lock.

One mechanism: every control of the section is in ``_CURATION_LOCK_WIDGETS``
and greys under ANY Step 3 operation; Start Training never renames itself;
``refinement_cancel_btn`` is the one training Cancel (live only while it
trains); Extract stays its own Cancel while ``_extraction_running``.  Run on
the REAL built panel, offscreen, one file at a time (``-o addopts=""``).
"""
from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication, QWidget

from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as Ctl,
)

#: The controls that did not lock before B2
LOCKED_CONTROLS = (
    "refine_patch_list", "refinement_mode_combo", "refine_init_combo",
    "refinement_source_combo", "refinement_source_combo_axial",
    "refine_pretrained_combo", "refine_pretrained_combo_axial",
    "refine_import_path", "refine_import_path_axial",
    "refine_import_browse_btn", "refine_import_browse_btn_axial",
    "refine_model_name", "refine_preset_combo", "refine_apply_preset_btn",
    "refine_epochs_sb", "refine_lr_sb", "refine_batch_sb",
    "refine_wd_sb", "refine_momentum_sb", "refine_min_masks_sb",
    "refine_chan1_combo", "refine_chan2_combo", "refine_seed_sb",
    "refine_normalize_cb", "refine_rescale_cb", "refine_sgd_cb",
    "refine_compute_flows_cb", "refine_device_gpu", "refine_device_cpu",
    "refine_train_frac_sb", "refine_chunk_size_sb", "refine_spatial_split_cb",
    "refine_select_all_btn", "refine_deselect_all_btn",
    "clear_extracted_pairs_btn", "refine_configure_per_orientation_btn",
    "refine_save_config_btn", "refine_load_config_btn",
    "launch_refinement_btn",
)


class _Shell:
    def lock_step_nav(self, locked, owner=None):
        pass


@pytest.fixture(scope="module")
def ui():
    app = QApplication.instance() or QApplication([])
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    yield w
    w.deleteLater()
    app.processEvents()


def _ctl(ui):
    c = Ctl.__new__(Ctl)
    c.ui = ui
    c._get_open_annotator_widget = lambda: None
    c._workflow_shell = _Shell()
    for flag in ("_curation_busy", "_patch_load_in_flight",
                 "_prediction_running", "_reconstruction_running",
                 "_refinement_running", "_extraction_running",
                 "_commit_triage_in_flight", "_save_in_flight",
                 "_batch_refine_running", "_extract_all_running",
                 "_add_mode_running", "_reshape_running", "_refine_running",
                 "_qc_extract_running", "_refinement_cancel_requested"):
        setattr(c, flag, False)
    c._refinement_queue = []
    c._refinement_launch_snapshot = None
    c._refine_pair_counts = (10, 10, 1)       # pairs on disk: Start is live
    return c


def test_every_control_is_in_the_one_lock_list(ui):
    listed = set(Ctl._CURATION_LOCK_WIDGETS)
    for name in LOCKED_CONTROLS:
        assert hasattr(ui, name), f"the panel lost {name}"
        assert name in listed, f"{name} is not in _CURATION_LOCK_WIDGETS"
    # the live ones are NOT in it
    assert "refinement_cancel_btn" not in listed
    assert "extract_training_pairs_btn" not in listed
    assert "refine_view_models_btn" not in listed


@pytest.mark.parametrize("flag", ["_refinement_running",
                                  "_prediction_running"])
def test_every_control_greys_under_any_operation(ui, flag):
    c = _ctl(ui)
    c._refresh_step3_lock()
    assert all(getattr(ui, n).isEnabled() for n in LOCKED_CONTROLS)
    setattr(c, flag, True)
    c._refresh_step3_lock()
    still_live = [n for n in LOCKED_CONTROLS
                  if getattr(ui, n).isEnabled()]
    assert still_live == [], still_live
    setattr(c, flag, False)
    c._refresh_step3_lock()
    assert all(getattr(ui, n).isEnabled() for n in LOCKED_CONTROLS)


def test_cancel_is_live_only_while_training(ui):
    c = _ctl(ui)
    c._refresh_step3_lock()
    assert ui.refinement_cancel_btn.isEnabled() is False      # idle
    c._prediction_running = True
    c._refresh_step3_lock()
    assert ui.refinement_cancel_btn.isEnabled() is False      # not ours
    c._prediction_running = False
    c._refinement_running = True
    c._refresh_step3_lock()
    assert ui.refinement_cancel_btn.isEnabled() is True       # training
    c._refinement_cancel_requested = True                     # asked
    c._refresh_step3_lock()
    assert ui.refinement_cancel_btn.isEnabled() is False
    c._refinement_running = False
    c._refinement_cancel_requested = False
    c._refresh_step3_lock()
    assert ui.refinement_cancel_btn.isEnabled() is False


def test_extract_stays_its_own_cancel_under_the_lock(ui):
    c = _ctl(ui)
    c._extraction_running = True
    c._refresh_step3_lock()
    assert ui.extract_training_pairs_btn.isEnabled() is True      # M10
    assert ui.launch_refinement_btn.isEnabled() is False
    assert ui.refine_patch_list.isEnabled() is False
    c._extraction_running = False
    c._refinement_running = True
    c._refresh_step3_lock()
    assert ui.extract_training_pairs_btn.isEnabled() is False     # training
    c._refinement_running = False
    c._refresh_step3_lock()


def test_the_lock_line_sentences(ui):
    c = _ctl(ui)
    c._refresh_step3_lock()
    assert ui.refine_lock_line.text() == ""
    # training: Model Refinement's own sentence, with the pass numbers
    c._refinement_running = True
    c._refinement_launch_snapshot = {"buckets": ["lateral", "axial"]}
    c._current_refinement_mode = "Fine-tune Lateral"
    c._refinement_queue = ["Fine-tune Axial"]
    c._refresh_step3_lock()
    assert ui.refine_lock_line.text() == M.MR_LOCK_TRAINING.format(
        bucket="lateral", i=1, n=2)
    c._current_refinement_mode = "Fine-tune Axial"
    c._refinement_queue = []
    c._refresh_step3_lock()
    assert ui.refine_lock_line.text() == M.MR_LOCK_TRAINING.format(
        bucket="axial", i=2, n=2)
    c._refinement_running = False
    # extraction
    c._extraction_running = True
    c._refresh_step3_lock()
    assert ui.refine_lock_line.text() == M.MR_LOCK_EXTRACT
    c._extraction_running = False
    # any other operation: the S3_LOCK_SENTENCES family
    c._prediction_running = True
    c._refresh_step3_lock()
    assert ui.refine_lock_line.text() == P.S3_LOCK_PREDICT
    c._prediction_running = False
    c._refresh_step3_lock()
    assert ui.refine_lock_line.text() == ""


def test_the_training_lock_sentence_no_longer_names_the_old_title():
    assert "Train a better model" not in P.S3_LOCK_TRAINING
    assert "better model" not in P.S3_LOCK_TRAINING


def test_start_training_never_renames_itself(ui):
    """I8: after ANY number of runs the idle button reads "Start Training"
    and "Cancel Training" / "Start Fine-Tuning" never appear."""
    from napari_epicycle.fusion_reconstruction import controller as cmod
    c = _ctl(ui)
    c._refresh_step3_lock = lambda: None
    c._refinement_launch_snapshot = {"buckets": ["lateral"]}
    for _ in range(3):
        c._refinement_running = True
        c._restore_launch_button()
        assert ui.launch_refinement_btn.text() == M.MR_START_BTN
    assert ui.launch_refinement_btn.text() == "Start Training"
    src = open(cmod.__file__, encoding="utf-8").read()
    assert "Cancel Training\")" not in src
    assert "Start Fine-Tuning" not in src


def test_cancel_button_is_wired_to_the_cancel_path():
    import inspect
    src = inspect.getsource(Ctl._connect_signals)
    assert "refinement_cancel_btn.clicked.connect(self._cancel_refinement)" \
        in src


def test_gates_grey_start_without_pairs_and_extract_without_a_tick(ui):
    """H5: on every refresh, not only on a click."""
    from napari_epicycle.fusion_reconstruction import _paths  # noqa: F401
    shown = []
    c = _ctl(ui)
    c._set_refinement_status = lambda t, k="warn": shown.append((t, k))
    c._load_extraction_manifest = lambda: {}
    c.working_dir = ""
    c._refine_pair_counts = (0, 0, 0)
    ui.set_refine_ticks(False)
    c._refresh_refine_gates(foot=True)
    assert ui.launch_refinement_btn.isEnabled() is False
    assert ui.extract_training_pairs_btn.isEnabled() is False
    assert shown[-1] == (M.MR_STATUS_NOTHING, "neutral")
    assert ui.refine_extraction_status_label.toolTip() == M.MR_EXTRACT_NEEDS_TICK  # the line is cut at two lines; the whole sentence is the tooltip
    c._refine_pair_counts = (30, 12, 2)
    c._refresh_refine_gates(foot=True)
    assert ui.launch_refinement_btn.isEnabled() is True
    assert shown[-1][0] == M.MR_STATUS_IDLE.format(
        which=M.MR_DONE_WHICH_BOTH, t="42") or shown[-1][0].startswith(
        "Ready · ")
    # never while an operation holds the lock
    c._step3_locked = True
    ui.launch_refinement_btn.setEnabled(False)
    c._refresh_refine_gates(foot=True)
    assert ui.launch_refinement_btn.isEnabled() is False


def test_few_pairs_block_follows_the_mode(ui):
    """F10: the lateral warning is about the lateral model only."""
    ui.refinement_mode_combo.setCurrentText("Fine-tune Axial")
    ui.set_refine_will_train(0, 80, 2)
    assert ui.refine_few_pairs_label.isHidden()
    ui.refinement_mode_combo.setCurrentText("Fine-tune Both")
    ui.set_refine_will_train(0, 80, 2)
    assert not ui.refine_few_pairs_label.isHidden()
    assert ui.refine_few_pairs_label.text() == M.MR_NO_LATERAL_PAIRS
    ui.set_refine_will_train(30, 80, 2)
    assert ui.refine_few_pairs_label.text() == M.MR_FEW_PAIRS_WARN.format(n=30)
    ui.set_refine_will_train(300, 80, 2)
    assert ui.refine_few_pairs_label.isHidden()
