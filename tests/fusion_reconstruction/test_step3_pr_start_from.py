"""Start from, its persistence, and the two inference changes of R2a.

* ONE writer, ``_apply_start_from(key)``; ``_apply_recon_defaults`` a
  wrapper; ``_USEG3D_DEFAULTS`` gone; no item writes a hidden knob.
* "This run's last reconstruction" reads THIS run's newest manifest,
  applies EpiCycle's defaults first and overlays the recorded keys; it is
  absent without a reconstruction; an older manifest shows the note.
* A hand edit flips the list to Custom; the key is persisted per run
  (``start_from`` in the run's block, not workspace-level) and read on
  workspace open and on a run switch.
* Compute masks is OFF: no ``_masks.npy``; the cache restore accepts a
  missing mask.  Rescale unticked passes ``None`` (Step 4's F25 rule).

A real panel + controller (``FusionReconstructionCuration(viewer=None)``).
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

HIDDEN = ("flow_thresh", "cellprob_thresh", "compute_masks_cb",
          "do_flow_remove_cb", "flow_threshold_sb", "fc_dtform_combo",
          "fc_edt_percentile_sb", "fc_power_dist_sb", "fc_smooth_skel_sb",
          "im_dtform_combo", "im_smooth_binary_sb", "im_smooth_skel_sb",
          "im_edt_percentile_sb", "im_power_dist_sb", "method_combo")


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
    # The panel warns with a modal box when the core spinner is set to
    # nearly every core of the machine.  The random hand edits below set it
    # to 1-3, which is "nearly every core" on a 4-core CI runner; offscreen,
    # nobody can press OK and the run hangs.  Record instead of showing.
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    widget = FusionReconstructionCuration(viewer=None)
    yield widget
    timer = getattr(widget.controller.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    widget.deleteLater()
    app.processEvents()


def _value(widget):
    if hasattr(widget, "currentIndex"):
        return ("combo", widget.currentIndex(), widget.currentText())
    if hasattr(widget, "value"):
        return ("spin", widget.value())
    return ("check", widget.isChecked())


def _hidden(ui):
    return {n: _value(getattr(ui, n)) for n in HIDDEN}


def _open(w, ws: Path):
    c = w.controller
    c.working_dir = str(ws)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(ws))
    return c


def _manifest(ws: Path, patch="P1", *, version=None, recon=None, seg=None,
              run_root=None):
    root = run_root or (ws / "artifacts" / "step3" / "runs")
    d = root / patch
    d.mkdir(parents=True, exist_ok=True)
    data = {"run_id": patch, "mode": "real",
            "seg_config": seg if seg is not None else {"channels": [3, 2]},
            "recon_config": recon if recon is not None else {
                "method": "direct", "skip_postprocess": False,
                "pre_smooth_probs": True,
                "gradient_descent": {"n_iter": 50, "momenta": 0.9,
                                     "gradient_decay": 0.002,
                                     "do_mp": False, "renorm_gradient": True,
                                     "tile_shape": [64, 128, 128],
                                     "tile_overlap_ratio": 0.1},
                "flow_consistency": {"do_flow_remove": True,
                                     "flow_threshold": 0.85}}}
    if version is not None:
        data["manifest_version"] = version
    (d / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    return d / "manifest.json"


def _items(combo):
    return [combo.itemData(i) for i in range(combo.count())]


# ---------------------------------------------------------------- one writer

def test_the_old_tables_and_lambdas_are_gone(w):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    assert not hasattr(FusionReconstructionController, "_USEG3D_DEFAULTS")
    assert w.useg3d_defaults_btn.receivers(w.useg3d_defaults_btn.clicked) == 0
    assert w.method_combo.receivers(w.method_combo.currentTextChanged) == 0


@pytest.mark.parametrize("key", ["epicycle", "useg3d", "last_run"])
def test_no_item_writes_a_hidden_knob(w, tmp_path, key):
    c = _open(w, tmp_path)
    _manifest(tmp_path)
    c._refresh_start_from_items()
    before = _hidden(w)
    c._apply_start_from(key, save=False)
    assert _hidden(w) == before


def test_epicycle_defaults_reset_every_shared_widget(w):
    from napari_epicycle.ui_shell._recon_presets import STEP4_PRESET_EPICYCLE
    c = w.controller
    w.batch_sb.setValue(8)
    w.gd_iters_sb.setValue(300)
    w.norm_cb.setChecked(False)
    w.diameter_sb.setValue(30)
    w.sf_max_size_sb.setValue(5000)
    c._apply_start_from("epicycle", save=False)
    assert w.batch_sb.value() == STEP4_PRESET_EPICYCLE["batch_size_sb"]
    assert w.gd_iters_sb.value() == 25
    assert w.norm_cb.isChecked()
    assert w.diameter_sb.value() == 0
    assert w.sf_max_size_sb.value() == -1
    assert c._start_from_key() == "epicycle"


def test_the_wrapper_and_the_hidden_button_use_the_one_writer(w):
    c = w.controller
    w.gd_iters_sb.setValue(300)
    assert c._start_from_key() == "custom"
    w.epicycle_defaults_btn.click()
    assert w.gd_iters_sb.value() == 25
    assert c._start_from_key() == "epicycle"
    calls = []
    c._apply_start_from = lambda key, **kw: calls.append(key)
    c._apply_recon_defaults("useg3d")
    c._apply_recon_defaults("epicycle")
    assert calls == ["useg3d", "epicycle"]


def test_useg3d_writes_its_own_defaults(w):
    from napari_epicycle.ui_shell._recon_presets import useg3d_preset
    preset = useg3d_preset()
    if not preset:
        pytest.skip("u-Segment3D is not importable here")
    c = w.controller
    c._apply_start_from("useg3d", save=False)
    assert w.gd_iters_sb.value() == preset["gd_iters_sb"]
    assert w.post_small_obj_sb.value() == preset["post_small_obj_sb"]
    assert w.ccg_ksize_sb.value() == preset["ccg_ksize_sb"]
    assert c._start_from_key() == "useg3d"


# ---------------------------------------------------------------- last run

def test_the_last_run_item_is_absent_without_a_reconstruction(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    c = _open(w, tmp_path)
    c._refresh_start_from_items()
    assert "last_run" not in _items(w.start_from_combo)
    assert w.pr_settings_desc.text() == P.S3_SETTINGS_DESC_NO_RUN
    _manifest(tmp_path)
    c._refresh_start_from_items()
    # the agreed order (the R2a fold):
    # EpiCycle · u-Segment3D · this run's last · Custom (after an edit)
    assert _items(w.start_from_combo) == ["epicycle", "useg3d", "last_run"]
    assert w.start_from_combo.itemText(2) == P.S3_PRESET_LAST_RUN
    assert w.pr_settings_desc.text() == P.S3_SETTINGS_DESC


def test_custom_comes_last_after_a_hand_edit(w, tmp_path):
    c = _open(w, tmp_path)
    _manifest(tmp_path)
    c._refresh_start_from_items()
    c._apply_start_from("epicycle", save=False)
    w.ccg_alpha_sb.setValue(2.0)
    assert _items(w.start_from_combo) == ["epicycle", "useg3d", "last_run",
                                          "custom"]
    # a rebuild while Custom is up keeps the order and Custom last
    c._refresh_start_from_items()
    assert _items(w.start_from_combo) == ["epicycle", "useg3d", "last_run",
                                          "custom"]
    assert c._start_from_key() == "custom"


def test_the_last_run_is_epicycle_then_the_recorded_keys(w, tmp_path):
    c = _open(w, tmp_path)
    _manifest(tmp_path)
    c._refresh_start_from_items()
    w.batch_sb.setValue(8)             # not recorded -> EpiCycle's 128
    w.ccp_ksize_sb.setValue(3)         # not recorded -> EpiCycle's 25
    c._apply_start_from("last_run", save=False)
    assert w.batch_sb.value() == 128
    assert w.ccp_ksize_sb.value() == 25
    assert w.gd_iters_sb.value() == 50                    # recorded
    assert w.gd_momenta_sb.value() == pytest.approx(0.9)
    assert not w.do_mp_cb.isChecked()
    assert w.gd_tile_z_sb.value() == 64
    assert not w.skip_postprocess_cb.isChecked()
    assert w.ps_smooth_probs_cb.isChecked()
    # the recorded Cellpose codes are Step 3's combo indexes
    assert w.chan1_combo.currentIndex() == 3
    assert w.chan2_combo.currentIndex() == 2
    # the recorded flow-consistency group never reaches a hidden widget
    assert not w.do_flow_remove_cb.isChecked()
    assert w.flow_threshold_sb.value() == pytest.approx(0.8)


def test_the_last_run_reads_only_this_runs_manifests(w, tmp_path):
    c = _open(w, tmp_path)
    # a NAMED run's reconstruction sits under runs/cells/reconstructions/
    named = tmp_path / "artifacts" / "step3" / "runs" / "cells" / \
        "reconstructions"
    _manifest(tmp_path, run_root=named)
    c._refresh_start_from_items()
    assert "last_run" not in _items(w.start_from_combo)


def test_the_newest_manifest_wins(w, tmp_path):
    c = _open(w, tmp_path)
    old = _manifest(tmp_path, "P1")
    new = _manifest(tmp_path, "P2", recon={"gradient_descent": {
        "n_iter": 77}})
    os.utime(old, (1_000_000, 1_000_000))
    c._refresh_start_from_items()
    c._apply_start_from("last_run", save=False)
    assert w.gd_iters_sb.value() == 77
    assert c._last_run_preset["path"] == str(new)


@pytest.mark.parametrize("version,shown", [(None, True), (2, False)])
def test_the_note_shows_for_an_older_manifest_only(w, tmp_path, version,
                                                   shown):
    c = _open(w, tmp_path)
    _manifest(tmp_path, version=version)
    c._refresh_start_from_items()
    c._apply_start_from("last_run", save=False)
    assert w.pr_last_run_note.isHidden() is (not shown)
    c._apply_start_from("epicycle", save=False)
    assert w.pr_last_run_note.isHidden()


def test_reset_to_predicted_drops_the_last_run_item(w, tmp_path):
    """Audit B MED-1: deleting the run's reconstruction (Reset to
    Predicted) must take "This run's last reconstruction" off the list."""
    c = _open(w, tmp_path)
    _manifest(tmp_path, "P1")
    c.state_manager.set_patch_status("P1", "reconstructed")
    c._refresh_start_from_items()
    assert "last_run" in _items(w.start_from_combo)
    assert c._reset_patches_to_predicted(["P1"]) == 1
    assert not (tmp_path / "artifacts" / "step3" / "runs" / "P1").exists()
    assert "last_run" not in _items(w.start_from_combo)
    assert w.pr_settings_desc.text() != ""


def test_a_stale_cached_last_run_item_never_writes(w, tmp_path):
    """The cached manifest is re-checked at the moment of applying: a
    reconstruction deleted behind the list's back drops the item, says so
    in the log and on screen, and writes nothing."""
    import shutil

    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    c = _open(w, tmp_path)
    _manifest(tmp_path, "P1")
    c._refresh_start_from_items()
    c._apply_start_from("epicycle")
    before = {n: _value(getattr(w, n)) for n in (
        "gd_iters_sb", "gd_momenta_sb", "ps_smooth_probs_cb", "chan1_combo",
        "chan2_combo", "gd_tile_z_sb")}
    shutil.rmtree(tmp_path / "artifacts" / "step3" / "runs" / "P1")
    logs = []
    c._append_log = logs.append
    idx = w.start_from_combo.findData("last_run")
    w.start_from_combo.setCurrentIndex(idx)     # what a pick does first
    w.start_from_combo.activated.emit(idx)
    after = {n: _value(getattr(w, n)) for n in before}
    assert after == before
    assert "last_run" not in _items(w.start_from_combo)
    assert c._start_from_key() == "epicycle"    # back where it was
    assert c.state_manager.get_start_from() == "epicycle"
    assert P.S3_LOG_LAST_RUN_VANISHED in logs
    assert not w.pr_last_run_note.isHidden()
    assert w.pr_last_run_note.text() == P.S3_LAST_RUN_VANISHED


def test_a_clamped_or_unknown_preset_value_is_logged(w, tmp_path):
    """Audit B LOW-1: a value Qt clamps, a method the list does not hold
    and a channel code the combo does not hold are each named in the log
    (S3_LOG_PRESET_INEXACT), never silently dropped."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    c = _open(w, tmp_path)
    _manifest(tmp_path, seg={"channels": [9, 1]})
    c._refresh_start_from_items()
    real = c._preset_values

    def fake(key):
        values = real(key)
        values.update({"gd_iters_sb": 5000, "ccp_alpha_sb": 0.123,
                       "method_combo": "indirect"})
        return values
    c._preset_values = fake
    logs = []
    c._append_log = logs.append
    c._apply_start_from("last_run", save=False)
    assert w.gd_iters_sb.value() == 1000                 # Qt clamped it
    assert w.ccp_alpha_sb.value() == pytest.approx(0.12)  # two decimals
    inexact = [m for m in logs if m.startswith("[SETTINGS] Start from: ")
               and "could not take" in m]
    named = {m.split(" ")[3] for m in inexact}
    assert {"gd_iters_sb", "ccp_alpha_sb", "method_combo",
            "chan1_combo"} <= named, inexact
    assert P.S3_LOG_PRESET_INEXACT.format(
        name="gd_iters_sb", wanted=5000, got=1000) in logs
    assert w.method_combo.currentData() == "direct"
    assert w.chan1_combo.currentIndex() == 2             # left as it was
    assert w.chan2_combo.currentIndex() == 1


def test_an_exact_preset_logs_nothing_inexact(w, tmp_path):
    c = _open(w, tmp_path)
    logs = []
    c._append_log = logs.append
    c._apply_start_from("epicycle", save=False)
    assert not [m for m in logs if "could not take" in m], logs


def test_useg3d_with_no_values_is_refused_on_screen(w, tmp_path,
                                                     monkeypatch):
    """Audit B LOW-3: with u-Segment3D not importable its preset resolves
    to nothing; the pick is refused with a sentence on screen and the list
    stays where it was -- it never claims a preset it did not apply."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    from napari_epicycle.fusion_reconstruction import controller as ctl
    monkeypatch.setattr(ctl, "useg3d_preset", lambda: {})
    c = _open(w, tmp_path)
    c._apply_start_from("epicycle")
    logs = []
    c._append_log = logs.append
    idx = w.start_from_combo.findData("useg3d")
    w.start_from_combo.setCurrentIndex(idx)     # what a pick does first
    w.start_from_combo.activated.emit(idx)
    assert c._start_from_key() == "epicycle"
    assert c.state_manager.get_start_from() == "epicycle"
    assert "useg3d" in _items(w.start_from_combo)    # still offered
    assert P.S3_LOG_USEG3D_UNAVAILABLE in logs
    assert not w.pr_last_run_note.isHidden()
    assert w.pr_last_run_note.text() == P.S3_USEG3D_UNAVAILABLE


def test_a_stored_useg3d_with_no_values_shows_custom(w, tmp_path,
                                                     monkeypatch):
    from napari_epicycle.fusion_reconstruction import controller as ctl
    monkeypatch.setattr(ctl, "useg3d_preset", lambda: {})
    _write_state(tmp_path, {"default": {"patches": {},
                                        "start_from": "useg3d"}})
    c = _open(w, tmp_path)
    w.gd_iters_sb.setValue(321)
    c._restore_start_from()
    assert c._start_from_key() == "custom"
    assert w.gd_iters_sb.value() == 321


# ---------------------------------------------------------------- Custom

def test_a_hand_edit_flips_to_custom_and_is_saved(w, tmp_path):
    c = _open(w, tmp_path)
    c._apply_start_from("epicycle")
    assert c.state_manager.get_start_from() == "epicycle"
    w.ccg_alpha_sb.setValue(2.0)
    assert c._start_from_key() == "custom"
    assert "custom" in _items(w.start_from_combo)
    assert c.state_manager.get_start_from() == "custom"
    # choosing an item again removes Custom
    c._apply_start_from("epicycle")
    assert "custom" not in _items(w.start_from_combo)


def test_a_preset_is_not_a_hand_edit(w):
    c = w.controller
    w.gd_iters_sb.setValue(300)
    c._apply_start_from("epicycle", save=False)
    assert c._start_from_key() == "epicycle"


def test_choosing_an_item_in_the_list_applies_it(w):
    c = w.controller
    w.gd_iters_sb.setValue(300)
    idx = w.start_from_combo.findData("epicycle")
    w.start_from_combo.activated.emit(idx)
    assert w.gd_iters_sb.value() == 25


def test_a_channel_pick_is_a_hand_edit(w, tmp_path):
    """The R2a fold (audit B MED-2 / A LOW-1): the two
    channel combos are settings, so picking one flips the list to Custom
    and saves "custom"."""
    c = _open(w, tmp_path)
    c._apply_start_from("epicycle")
    w.chan1_combo.setCurrentIndex(1)
    assert c._start_from_key() == "custom"
    assert c.state_manager.get_start_from() == "custom"
    c._apply_start_from("epicycle")
    w.chan2_combo.setCurrentIndex(3)
    assert c._start_from_key() == "custom"


def test_the_bundle_channel_sync_is_not_a_hand_edit(w, tmp_path):
    """A model pick writes the bundle's training channels into the combos
    (``[CHANNEL-SYNC]``); that is the program, not the user, so the list
    stays where it was."""
    c = _open(w, tmp_path)
    c._apply_start_from("epicycle")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "lateral_training_config.json").write_text(json.dumps(
        {"hyperparameters": {"channels": [3, 0]}}), encoding="utf-8")
    c._try_sync_prediction_channels_from_bundle(bundle)
    assert w.chan1_combo.currentIndex() == 3
    assert w.chan2_combo.currentIndex() == 0
    assert c._start_from_key() == "epicycle"
    assert c.state_manager.get_start_from() == "epicycle"
    assert not getattr(c, "_applying_preset", False)


def test_the_last_run_channels_are_not_a_hand_edit(w, tmp_path):
    c = _open(w, tmp_path)
    _manifest(tmp_path, seg={"channels": [1, 3]})
    c._refresh_start_from_items()
    c._apply_start_from("last_run", save=False)
    assert w.chan1_combo.currentIndex() == 1
    assert w.chan2_combo.currentIndex() == 3
    assert c._start_from_key() == "last_run"


# ---------------------------------------------------------------- persistence

def test_start_from_is_a_per_run_key(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _migrate_step3_runs as M
    assert "start_from" not in M.WORKSPACE_LEVEL_KEYS_V1
    c = _open(w, tmp_path)
    c.state_manager.set_start_from("useg3d")
    c.state_manager.flush()
    c.state_manager.save()
    data = json.loads((tmp_path / c.state_manager.STATE_FILENAME).read_text(
        encoding="utf-8"))
    assert "start_from" not in data
    assert data["runs"]["default"]["start_from"] == "useg3d"


def test_a_missing_key_is_epicycle(w, tmp_path):
    c = _open(w, tmp_path)
    assert c.state_manager.get_start_from() == "epicycle"
    with pytest.raises(ValueError):
        c.state_manager.set_start_from("bogus")


def _write_state(ws: Path, runs: dict, active="default"):
    from napari_epicycle.fusion_reconstruction import _paths
    (ws / _paths.STEP3_STATE_FILENAME).write_text(json.dumps({
        "schema_version": 3, "active_run_id": active, "runs": runs}),
        encoding="utf-8")


def test_the_key_is_read_on_workspace_open(w, tmp_path):
    from napari_epicycle.ui_shell._recon_presets import useg3d_preset
    if not useg3d_preset():
        pytest.skip("u-Segment3D is not importable here")
    _write_state(tmp_path, {"default": {"patches": {},
                                        "start_from": "useg3d"}})
    c = _open(w, tmp_path)
    c._restore_start_from()
    assert c._start_from_key() == "useg3d"
    assert w.gd_iters_sb.value() == useg3d_preset()["gd_iters_sb"]


def test_custom_is_not_restored_the_widgets_keep_their_values(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    _write_state(tmp_path, {"default": {"patches": {},
                                        "start_from": "custom"}})
    c = _open(w, tmp_path)
    w.gd_iters_sb.setValue(321)
    logs = []
    c._append_log = logs.append
    c._restore_start_from()
    assert c._start_from_key() == "custom"
    assert w.gd_iters_sb.value() == 321
    assert P.S3_LOG_CUSTOM_NOT_RESTORED in logs


def test_a_gone_last_run_shows_custom(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    _write_state(tmp_path, {"default": {"patches": {},
                                        "start_from": "last_run"}})
    c = _open(w, tmp_path)
    logs = []
    c._append_log = logs.append
    c._restore_start_from()
    assert c._start_from_key() == "custom"
    assert P.S3_LOG_LAST_RUN_GONE in logs


def test_a_run_switch_reads_the_new_runs_key(w, tmp_path):
    from napari_epicycle.ui_shell._recon_presets import useg3d_preset
    if not useg3d_preset():
        pytest.skip("u-Segment3D is not importable here")
    _write_state(tmp_path, {
        "default": {"patches": {}, "start_from": "epicycle"},
        "cells": {"patches": {}, "start_from": "useg3d",
                  "display_name": "cells"}})
    c = _open(w, tmp_path)
    c._restore_start_from()
    assert c._start_from_key() == "epicycle"
    c._switch_active_run("cells")
    assert c.state_manager.get_active_run_id() == "cells"
    assert c._start_from_key() == "useg3d"
    w.gd_iters_sb.setValue(222)                       # Custom in "cells"
    c._switch_active_run("default")
    assert c._start_from_key() == "epicycle"
    assert w.gd_iters_sb.value() == 25


def test_a_switch_to_a_run_with_no_key_keeps_the_values_as_custom(w,
                                                                 tmp_path):
    """The R2a fold: a run that never stored a choice
    is not given EpiCycle's defaults on a switch -- the widgets keep their
    values, the list says Custom, and nothing is written for that run."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    _write_state(tmp_path, {
        "default": {"patches": {}, "start_from": "epicycle"},
        "cells": {"patches": {}, "display_name": "cells"}})
    c = _open(w, tmp_path)
    c._restore_start_from()
    assert c._start_from_key() == "epicycle"
    c._applying_preset = True           # a value no preset holds, not
    w.gd_iters_sb.setValue(222)         # a hand edit (nothing saved)
    c._applying_preset = False
    logs = []
    c._append_log = logs.append
    c._switch_active_run("cells")
    assert c.state_manager.get_active_run_id() == "cells"
    assert c._start_from_key() == "custom"
    assert w.gd_iters_sb.value() == 222
    assert c.state_manager.stored_start_from() is None
    assert P.S3_LOG_RUN_NO_CHOICE in logs


def test_a_workspace_open_with_no_key_shows_epicycle(w, tmp_path):
    """"missing = epicycle" is the workspace-open rule: on a
    fresh open the widgets are untouched, so applying it changes
    nothing, and the label is EpiCycle defaults.  Nothing is written."""
    _write_state(tmp_path, {"default": {"patches": {}}})
    c = _open(w, tmp_path)
    c._restore_start_from()
    assert c._start_from_key() == "epicycle"
    assert w.gd_iters_sb.value() == 25
    assert c.state_manager.stored_start_from() is None


def test_the_three_restore_sites_call_it():
    """The run switch, the layout conversion and the workspace open each
    read the key right after the model selection is restored."""
    import inspect

    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    src = inspect.getsource(C)
    assert src.count("self._restore_start_from(") == 3
    # only the run switch keeps a run without a stored key as Custom
    assert src.count("self._restore_start_from(on_switch=True)") == 1


# ---------------------------------------------------------------- inference

def test_compute_masks_is_off(w):
    cfg = w.controller._collect_seg_config()
    assert cfg.compute_masks is False


@pytest.mark.parametrize("ticked,expected", [(True, True), (False, None)])
def test_rescale_unticked_is_none(w, ticked, expected):
    w.rescale_cb.setChecked(ticked)
    assert w.controller._collect_seg_config().rescale is expected


def test_the_method_is_the_items_data(w):
    assert w.controller._collect_recon_config().method == "direct"


def test_no_masks_file_is_written_and_the_cache_restores(w, tmp_path):
    c = _open(w, tmp_path)
    flows = np.zeros((2, 3, 4, 4), np.float32)
    probs = np.zeros((3, 4, 4), np.float32)
    views = {v: (flows, probs, None) for v in ("xy", "xz", "yz")}
    c._persist_prediction_cache(views, patch_name="P1.tif")
    cache = c._prediction_cache_dir()
    names = sorted(p.name for p in cache.iterdir())
    assert not any("_masks" in n for n in names)
    assert len(names) == 6
    back = c._try_restore_prediction_cache("P1.tif")
    assert back is not None and set(back) == {"xy", "xz", "yz"}
    assert all(v[2] is None for v in back.values())


def test_segment_view_skips_cellposes_mask_step():
    """The masks flag reaches Cellpose: with ``compute_masks=False`` the
    model is asked for no masks and the view returns none."""
    from packages.epicycle_reconstruction.logic import _segment_view
    from packages.epicycle_training.config import SegmentationConfig

    seen = []

    class Model:
        def eval(self, img, **kw):
            seen.append(kw["compute_masks"])
            h, w_ = img.shape[-2:]
            return (np.zeros((h, w_), np.int32),
                    [None, np.zeros((2, h, w_), np.float32),
                     np.zeros((h, w_), np.float32)], None)

    image = np.zeros((3, 1, 4, 5), np.float32)
    cfg = SegmentationConfig(compute_masks=False)
    flows, probs, masks = _segment_view(image, Model(), "xy", cfg, None)
    assert masks is None and seen and not any(seen)
    assert flows.shape == (2, 3, 4, 5)


# ---------------------------------------------------------------- R2c: exact on v2

def _visible_randomise(ui, seed):
    """Every visible setting (the hidden knobs keep their built values; the
    CPU cores stay low so no high-core warning opens)."""
    import random

    from qtpy.QtWidgets import QAbstractSpinBox, QCheckBox, QDoubleSpinBox

    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    rng = random.Random(seed)
    for name in sorted(P.STEP3_SETTING_KEYS):
        if name in HIDDEN:
            continue
        w = getattr(ui, name)
        if name == "n_cpu_sb":
            w.setValue(rng.randint(1, 3))
        elif isinstance(w, QCheckBox):
            w.setChecked(rng.random() < 0.5)
        elif isinstance(w, QDoubleSpinBox):
            steps = int(round((w.maximum() - w.minimum()) / w.singleStep()))
            w.setValue(w.minimum() + rng.randint(0, max(steps, 1))
                       * w.singleStep())
        elif isinstance(w, QAbstractSpinBox):
            w.setValue(rng.randint(int(w.minimum()), int(w.maximum())))
    ui.chan1_combo.setCurrentIndex(rng.randrange(ui.chan1_combo.count()))
    ui.chan2_combo.setCurrentIndex(rng.randrange(ui.chan2_combo.count()))


def _write_v2(ws, c, w):
    """A v2 manifest of the panel as it is now, as Step 3's Reconstruct
    writes it (no prediction manifest: today's settings + the µm typed)."""
    from packages.epicycle_reconstruction.logic import build_real_manifest
    seg, extra = c._recon_recorded_seg("P1.tif", c._collect_seg_config(),
                                       w.diameter_sb.value())
    m = build_real_manifest(
        run_id="r", source_stack_path=str(ws / "P1.tif"),
        labels_path="l", labels_shape=(1, 1, 1), num_cells_raw=1,
        num_cells_final=1, axes="CZYX", seg_config=seg,
        recon_config=c._collect_recon_config(), extra_manifest=extra)
    d = ws / "artifacts" / "step3" / "runs" / "P1.tif"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return m


def _configs(c):
    import dataclasses
    return (dataclasses.asdict(c._collect_seg_config()),
            dataclasses.asdict(c._collect_recon_config()))


@pytest.mark.parametrize("seed", range(6))
def test_the_last_run_is_exact_on_a_v2_manifest(w, tmp_path, seed):
    """R2c: a v2 manifest records every setting, so the
    item gives back exactly the configs that reconstruction recorded --
    the inference fields and the CPU cores included -- and no note."""
    c = _open(w, tmp_path)
    _visible_randomise(w, seed)
    want = _configs(c)
    _write_v2(tmp_path, c, w)
    _visible_randomise(w, seed + 100)
    assert _configs(c) != want
    c._refresh_start_from_items()
    c._apply_start_from("last_run", save=False)
    assert _configs(c) == want
    assert w.pr_last_run_note.isHidden()


def test_the_inference_fields_need_a_v2_manifest(w, tmp_path):
    """A v1 manifest (no ``manifest_version``) never sets them: they stay
    EpiCycle's, and the note says so."""
    c = _open(w, tmp_path)
    _manifest(tmp_path, seg={"channels": [3, 2], "batch_size": 8,
                             "normalize": False, "rescale": None},
              recon={"n_cpu": 2, "method": "direct"})
    c._refresh_start_from_items()
    w.batch_sb.setValue(16)
    c._apply_start_from("last_run", save=False)
    assert w.batch_sb.value() == 128
    assert w.norm_cb.isChecked() and w.rescale_cb.isChecked()
    assert w.n_cpu_sb.value() == 6
    assert not w.pr_last_run_note.isHidden()


def test_a_v2_manifest_sets_the_inference_fields(w, tmp_path):
    c = _open(w, tmp_path)
    _manifest(tmp_path, version=2,
              seg={"channels": [1, 0], "batch_size": 8, "normalize": False,
                   "rescale": None, "diameter": 14.0},
              recon={"n_cpu": 2, "method": "direct"})
    d = tmp_path / "artifacts" / "step3" / "runs" / "P1"
    data = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    data["cell_width_um"] = 7.5
    (d / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    c._refresh_start_from_items()
    c._apply_start_from("last_run", save=False)
    assert w.batch_sb.value() == 8
    assert not w.norm_cb.isChecked()
    assert not w.rescale_cb.isChecked()          # None = the model's size
    assert w.n_cpu_sb.value() == 2
    assert w.diameter_sb.value() == pytest.approx(7.5)   # the µm, not px
    assert w.chan1_combo.currentIndex() == 1
    assert w.pr_last_run_note.isHidden()


def test_a_pixel_width_without_its_um_is_named_not_applied(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    c = _open(w, tmp_path)
    _manifest(tmp_path, version=2, seg={"channels": [2, 1], "diameter": 30.0},
              recon={"n_cpu": 2})
    c._refresh_start_from_items()
    w.diameter_sb.setValue(4.0)
    c._apply_start_from("last_run", save=False)
    assert w.diameter_sb.value() == 0              # EpiCycle's Auto
    head = P.S3_LOG_PRESET_INEXACT.split("{")[0]
    assert head in w.log_output.toPlainText()
    assert "diameter_sb" in w.log_output.toPlainText()


# ---------------------------------------------------------------- R2c: what Reconstruct records

def test_reconstruct_records_the_predictions_settings(w, tmp_path):
    from packages.epicycle_training.config import SegmentationConfig

    from napari_epicycle.fusion_reconstruction import _step3_manifests as S
    c = _open(w, tmp_path)
    used = SegmentationConfig(channels=[3, 0], diameter=21.0, rescale=None,
                              batch_size=16, normalize=False,
                              compute_masks=False)
    S.write_json_atomic(
        S.prediction_manifest_path(c._prediction_cache_dir(), "P1.tif"),
        S.build_prediction_manifest(patch_name="P1.tif", axes="ZYX",
                                    seg_config=used, cell_width_um=10.5))
    now = c._collect_seg_config()
    seg, extra = c._recon_recorded_seg("P1.tif", now, 3.0)
    assert seg == used
    # R6 adds the prediction manifest's workspace-relative path (and the
    # provenance block when the manifest holds one) to the same extras.
    assert extra["cell_width_um"] == 10.5
    assert extra["seg_config_source"] == "prediction"
    assert extra["prediction_manifest"].endswith("P1_prediction_manifest.json")
    # no prediction manifest (a cache made before R2c): today's settings
    seg, extra = c._recon_recorded_seg("P2.tif", now, 3.0)
    assert seg == _s3_with_diameter(now, None)
    assert extra == {"cell_width_um": 3.0, "seg_config_source": "settings",
                     "diameter_from": "settings"}


def _s3_with_diameter(seg, diameter):
    from napari_epicycle.fusion_reconstruction import _step3_manifests as S
    return S.with_diameter(seg, diameter)


def test_the_settings_record_puts_no_um_in_the_pixel_diameter(w, tmp_path):
    """R2c audit LOW-2: with no prediction manifest the record is today's
    settings, whose ``diameter`` is the µm number typed in the box --
    never recorded under Cellpose's pixel-named ``seg_config.diameter``;
    the µm lives in ``cell_width_um`` and ``diameter_from`` says where the
    record came from.  The config given is not changed."""
    c = _open(w, tmp_path)
    w.diameter_sb.setValue(12.5)
    now = c._collect_seg_config()
    assert now.diameter == 12.5
    seg, extra = c._recon_recorded_seg("P2.tif", now, 12.5)
    assert seg.diameter is None
    assert extra["cell_width_um"] == 12.5
    assert extra["diameter_from"] == "settings"
    assert now.diameter == 12.5


def test_reconstruct_passes_the_record_to_run_real(w, tmp_path):
    c = _open(w, tmp_path)
    c._try_restore_prediction_cache = lambda name: {"xy": 1}
    seen = {}
    c.reconstruction_service.run_real = lambda **kw: seen.update(kw) or {}
    c._reconstruct_single_patch(
        patch_name="P1.tif", stack_path="s", working_dir=str(tmp_path),
        lateral_path="l", axial_path="a",
        seg_config=c._collect_seg_config(),
        recon_config=c._collect_recon_config(), cell_width_um=2.5)
    assert seen["extra_manifest"] == {"cell_width_um": 2.5,
                                      "seg_config_source": "settings",
                                      "diameter_from": "settings"}
