"""A new Step 3 run starts clean, and each run remembers its own channels.

* A new run never carries the previous run's channel codes or "Remove
  specks smaller than": it starts from EpiCycle's defaults for ITS model.
  A Cellpose built-in records no channels, so "Channel to segment" starts
  blank and the helper at 0; a model whose training config is copied with
  the run fills both boxes.  The built-in nuclei model's specks are 100,
  every other model's 10,000.
* Each run remembers its two channel codes and its specks (``channels`` /
  ``specks`` in its block of ``step3_state.json``), written on every
  change, by hand or by the program, but NEVER while a run switch is in
  flight or while the run's memory is being put back.
* The memory is put back at the end of every Start from restore: the
  channels always; the specks only for Custom or no stored choice.
* Under EpiCycle defaults, picking another model puts the specks back to
  that model's value.

A real panel + controller (``FusionReconstructionCuration(viewer=None)``).
Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


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


def _write_state(ws: Path, runs: dict, active="default"):
    from napari_epicycle.fusion_reconstruction import _paths
    (ws / _paths.STEP3_STATE_FILENAME).write_text(json.dumps({
        "schema_version": 3, "active_run_id": active, "runs": runs}),
        encoding="utf-8")


def _read_runs(ws: Path) -> dict:
    from napari_epicycle.fusion_reconstruction import _paths
    data = json.loads((ws / _paths.STEP3_STATE_FILENAME).read_text(
        encoding="utf-8"))
    return data["runs"]


def _open(w, ws: Path, runs=None):
    if runs is not None:
        _write_state(ws, runs)
    c = w.controller
    c.working_dir = str(ws)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(ws))
    c._restore_start_from()
    return c


def _boxes(w):
    return (w.chan1_combo.currentIndex(), w.chan2_combo.currentIndex(),
            w.post_small_obj_sb.value())


def _hand(w, c1=None, c2=None, specks=None):
    """A user's edits (no program guard set)."""
    if c1 is not None:
        w.chan1_combo.setCurrentIndex(c1)
    if c2 is not None:
        w.chan2_combo.setCurrentIndex(c2)
    if specks is not None:
        w.post_small_obj_sb.setValue(specks)


class _Dialog:
    result = None

    def __init__(self, *a, **k):
        pass

    def exec_(self):
        return True

    def get_result(self):
        return dict(_Dialog.result)


def _new_run(c, monkeypatch, slug, init_spec, *, activate=True):
    from napari_epicycle.fusion_reconstruction import _new_run_dialog
    _Dialog.result = {"slug": slug, "display_name": slug, "target": "",
                      "description": "", "backend": "cellpose",
                      "init_spec": init_spec, "notes": "",
                      "activate": activate}
    monkeypatch.setattr(_new_run_dialog, "NewRunDialog", _Dialog)
    logs = []
    c._append_log = logs.append
    c._show_run_error = lambda title, body: logs.append(f"ERR {body}")
    c._on_new_run_clicked()
    return logs


def _pretrained(name):
    return {"mode": "pretrained", "pretrained_name": name}


def _trained_model(folder: Path, codes) -> Path:
    """A model file with its Step 2 training config beside it."""
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "Cell_lateral_round_01"
    weights.write_bytes(b"weights")
    (folder / f"{weights.name}_training_config.json").write_text(
        json.dumps({"hyperparameters": {"channels": list(codes)}}),
        encoding="utf-8")
    return weights


A_CUSTOM = {"patches": {}, "start_from": "custom"}


# ------------------------------------------------- a new run starts clean (I3)

@pytest.mark.parametrize("activate", [True, False])
def test_a_new_nuclei_run_never_carries_the_previous_runs_values(
        w, tmp_path, monkeypatch, activate):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _hand(w, c1=3, c2=2, specks=777)
    assert c._start_from_key() == "custom"
    logs = _new_run(c, monkeypatch, "nuc", _pretrained("nuclei"),
                    activate=activate)
    assert not [m for m in logs if m.startswith("ERR")], logs
    if not activate:
        assert c.state_manager.get_active_run_id() == "default"
        c._switch_active_run("nuc")
    assert c.state_manager.get_active_run_id() == "nuc"
    assert w.lateral_model_path.text() == "nuclei"
    assert _boxes(w) == (-1, 0, 100)
    assert w.chan1_combo.placeholderText() == P.S3_CHAN1_PLACEHOLDER
    assert c._start_from_key() == "epicycle"
    assert c.state_manager.stored_start_from() == "epicycle"
    assert c.state_manager.get_run_channels() == [None, 0]
    assert c.state_manager.get_run_specks() == 100
    assert P.S3_LOG_NEW_RUN_PICK.format(run="nuc") in logs
    assert logs.count(P.S3_LOG_NUCLEI_SPECKS.format(n=100)) == 1


def test_a_new_cell_built_in_run_is_blank_with_the_cell_specks(
        w, tmp_path, monkeypatch):
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _hand(w, c1=3, c2=2, specks=777)
    _new_run(c, monkeypatch, "cyto", _pretrained("cyto3"))
    assert _boxes(w) == (-1, 0, 10000)
    assert c._start_from_key() == "epicycle"


def test_a_new_run_from_a_trained_model_takes_its_channels(
        w, tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    from napari_epicycle.ui_shell._model_files import read_trained_channels
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _hand(w, c1=3, c2=2, specks=777)
    src = _trained_model(tmp_path / "models" / "m1", [1, 0])
    logs = _new_run(c, monkeypatch, "fromfile",
                    {"mode": "browse", "external_path": str(src)})
    assert not [m for m in logs if m.startswith("ERR")], logs
    lat = Path(w.lateral_model_path.text())
    assert lat.parent.name == "init_model"           # the run's own copy
    assert (lat.parent / "lateral_training_config.json").is_file()
    assert read_trained_channels(lat) == [1, 0]
    assert _boxes(w) == (1, 0, 10000)
    assert c.state_manager.get_run_channels() == [1, 0]
    assert P.S3_LOG_NEW_RUN_CHANNELS.format(
        run="fromfile", seg="1: red", helper="0: none") in logs


def test_the_new_run_seed_reads_no_file_for_a_built_in_name(w, tmp_path,
                                                             monkeypatch):
    """A built-in NAME is not a path: nothing beside the working folder is
    read for it, even when a training config sits there."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lateral_training_config.json").write_text(
        json.dumps({"hyperparameters": {"channels": [3, 3]}}),
        encoding="utf-8")
    c = w.controller
    assert c._new_run_seed({"lateral_model_path": "nuclei",
                            "axial_model_path": "nuclei"}) == (None, 0, 100)


def test_two_views_trained_on_different_codes_leave_the_box_blank():
    from napari_epicycle.fusion_reconstruction._step3_channels import (
        seed_codes,
    )
    assert seed_codes([2, 1], [2, 1]) == (2, 1)
    assert seed_codes([2, 1], [1, 0]) == (None, 0)
    assert seed_codes(None, None) == (None, 0)
    assert seed_codes([2, 1], None) == (None, 0)
    assert seed_codes([5, 1], [5, 1]) == (None, 0)
    assert seed_codes([True, 0], [True, 0]) == (None, 0)


# ------------------------------------------------------ the memory (I7, I7b)

def test_the_memory_round_trips_across_a_switch_and_a_reopen(w, tmp_path):
    c = _open(w, tmp_path, {
        "default": dict(A_CUSTOM),
        "b": {"patches": {}, "start_from": "custom", "display_name": "b",
              "channels": [3, 2], "specks": 500}})
    _hand(w, c1=1, c2=0, specks=300)
    assert c.state_manager.get_run_channels() == [1, 0]
    assert c.state_manager.get_run_specks() == 300
    c._switch_active_run("b")
    assert _boxes(w) == (3, 2, 500)
    c._switch_active_run("default")
    assert _boxes(w) == (1, 0, 300)
    assert c._start_from_key() == "custom"
    c.state_manager.flush()
    # reopen: build defaults first, then the run's memory
    w.chan1_combo.setCurrentIndex(2)
    w.chan2_combo.setCurrentIndex(1)
    c = _open(w, tmp_path)
    assert _boxes(w) == (1, 0, 300)
    assert c._start_from_key() == "custom"


def test_nothing_is_remembered_while_a_switch_is_in_flight(w, tmp_path):
    c = _open(w, tmp_path, {"default": {
        "patches": {}, "start_from": "custom",
        "channels": [1, 0], "specks": 300}})
    c._run_switch_in_flight = True
    try:
        _hand(w, c1=3, c2=2, specks=999)
    finally:
        c._run_switch_in_flight = False
    assert c.state_manager.get_run_channels() == [1, 0]
    assert c.state_manager.get_run_specks() == 300
    c._restoring_run_memory = True
    try:
        _hand(w, c1=2, c2=3, specks=888)
    finally:
        c._restoring_run_memory = False
    assert c.state_manager.get_run_channels() == [1, 0]
    assert c.state_manager.get_run_specks() == 300


def test_a_switch_back_and_forth_keeps_each_runs_memory_byte_for_byte(
        w, tmp_path):
    runs = {
        "default": {"patches": {}, "start_from": "custom",
                    "channels": [1, 0], "specks": 300},
        "b": {"patches": {}, "start_from": "custom", "display_name": "b",
              "channels": [3, 2], "specks": 500}}
    c = _open(w, tmp_path, runs)
    for target in ("b", "default", "b", "default"):
        c._switch_active_run(target)
    c.state_manager.flush()
    got = _read_runs(tmp_path)
    for slug in ("default", "b"):
        assert got[slug]["channels"] == runs[slug]["channels"], slug
        assert got[slug]["specks"] == runs[slug]["specks"], slug
    assert _boxes(w) == (1, 0, 300)


def test_the_memory_follows_a_program_write(w, tmp_path, monkeypatch):
    """The bundle sync is the program: the list stays EpiCycle defaults,
    and the run remembers the channels it wrote."""
    c = _open(w, tmp_path, {"default": {"patches": {},
                                        "start_from": "epicycle"}})
    monkeypatch.setattr(c, "_read_bundle_training_channels",
                        lambda d: [3, 1])
    c._try_sync_prediction_channels_from_bundle(str(tmp_path))
    assert _boxes(w)[:2] == (3, 1)
    assert c._start_from_key() == "epicycle"
    assert c.state_manager.get_run_channels() == [3, 1]


def test_the_specks_memory_waits_under_a_preset(w, tmp_path):
    """Under EpiCycle defaults the preset is the truth for the specks; the
    channels memory applies for every choice."""
    c = _open(w, tmp_path, {"default": {
        "patches": {}, "start_from": "epicycle",
        "channels": [1, 0], "specks": 300}})
    assert _boxes(w) == (1, 0, 10000)
    assert c._start_from_key() == "epicycle"


def test_a_run_with_no_choice_takes_its_memory_on_a_switch(w, tmp_path):
    c = _open(w, tmp_path, {
        "default": {"patches": {}, "start_from": "epicycle"},
        "old": {"patches": {}, "display_name": "old",
                "channels": [1, 0], "specks": 250}})
    c._switch_active_run("old")
    assert _boxes(w) == (1, 0, 250)
    assert c._start_from_key() == "custom"
    assert c.state_manager.stored_start_from() is None


def test_the_keys_survive_a_switch_round_trip_on_disk(w, tmp_path):
    c = _open(w, tmp_path, {
        "default": dict(A_CUSTOM),
        "b": {"patches": {}, "display_name": "b", "start_from": "custom"}})
    _hand(w, c1=1, c2=0, specks=321)
    c._switch_active_run("b")
    c.state_manager.flush()
    got = _read_runs(tmp_path)["default"]
    assert got["channels"] == [1, 0]
    assert got["specks"] == 321
    assert got["start_from"] == "custom"


@pytest.mark.parametrize("bad", [[4, 0], [0, None], ["1", 0], [1], [1, 2, 3],
                                 [True, 0], None])
def test_the_channel_memory_rejects_what_it_cannot_mean(w, tmp_path, bad):
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    with pytest.raises(ValueError):
        c.state_manager.set_run_channels(bad)


@pytest.mark.parametrize("bad", [-1, 1.5, "100", True, None])
def test_the_specks_memory_rejects_what_it_cannot_mean(w, tmp_path, bad):
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    with pytest.raises(ValueError):
        c.state_manager.set_run_specks(bad)


def test_an_unreadable_stored_memory_reads_as_none(w, tmp_path):
    c = _open(w, tmp_path, {"default": {"patches": {}, "start_from": "custom",
                                        "channels": "red", "specks": -5}})
    assert c.state_manager.get_run_channels() is None
    assert c.state_manager.get_run_specks() is None


# --------------------------------- a run made before runs remembered (H1)

@pytest.mark.parametrize("stored", ["epicycle", "custom", None])
def test_an_older_run_keeps_what_it_was_last_used_with(w, tmp_path,
                                                       monkeypatch, stored):
    """A run with no memory (made before runs remembered their channels)
    gets its current boxes and specks written as it is LEFT; a switch back
    gives them back, not the other run's."""
    run = {"patches": {}, "lateral_model_path": "cyto3",
           "axial_model_path": "cyto3", "model_source_mode": "pretrained"}
    if stored:
        run["start_from"] = stored
    c = _open(w, tmp_path, {"default": run})
    c._restore_persisted_model_selection()
    c._restore_start_from()
    before = _boxes(w)
    assert c.state_manager.get_run_channels() is None
    _new_run(c, monkeypatch, "nuc", _pretrained("nuclei"))
    w.chan1_combo.setCurrentIndex(1)                 # the pick: 1: red
    c._switch_active_run("default")
    assert _boxes(w) == before
    assert c.state_manager.get_run_channels() == [before[0], before[1]]
    c._switch_active_run("nuc")
    assert _boxes(w) == (1, 0, 100)


def test_the_outgoing_seed_never_overwrites_a_stored_memory(w, tmp_path):
    c = _open(w, tmp_path, {
        "default": {"patches": {}, "start_from": "custom",
                    "channels": [1, 0], "specks": 300},
        "b": {"patches": {}, "display_name": "b", "start_from": "custom"}})
    c._run_switch_in_flight = True               # values no hand set
    w.chan1_combo.setCurrentIndex(3)
    w.post_small_obj_sb.setValue(999)
    c._run_switch_in_flight = False
    c._seed_outgoing_run_memory()
    assert c.state_manager.get_run_channels() == [1, 0]
    assert c.state_manager.get_run_specks() == 300


# ------------------------------- filling the blank box is not Custom (H2)

def test_filling_the_blank_box_keeps_epicycle_defaults(w, tmp_path,
                                                       monkeypatch):
    """The pick a built-in run asks for keeps the list on EpiCycle
    defaults, so the specks still follow a model change; a later change
    of the filled box flips to Custom as before."""
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _new_run(c, monkeypatch, "cyto", _pretrained("cyto3"))
    assert _boxes(w) == (-1, 0, 10000)
    w.chan1_combo.setCurrentIndex(2)                 # the pick: 2: green
    assert c._start_from_key() == "epicycle"
    assert c.state_manager.stored_start_from() == "epicycle"
    assert c.state_manager.get_run_channels() == [2, 0]
    _pick(c, w, "nuclei")                            # Change… / Use
    assert w.post_small_obj_sb.value() == 100
    w.chan1_combo.setCurrentIndex(1)                 # a filled box changed
    assert c._start_from_key() == "custom"


# ----------------------------------------------- the model overlay (M6)

def _pick(c, w, name):
    w.lateral_model_path.setText(name)
    w.axial_model_path.setText(name)
    c._persist_current_model_selection(mode="pretrained")


def test_epicycle_defaults_give_the_nuclei_model_its_specks(w, tmp_path):
    c = _open(w, tmp_path, {"default": {"patches": {},
                                        "start_from": "epicycle"}})
    w.lateral_model_path.setText("nuclei")
    c._apply_start_from("epicycle", save=False)
    assert w.post_small_obj_sb.value() == 100
    w.lateral_model_path.setText("cyto3")
    c._apply_start_from("epicycle", save=False)
    assert w.post_small_obj_sb.value() == 10000


def test_another_model_picked_puts_the_specks_back(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    c = _open(w, tmp_path, {"default": {"patches": {},
                                        "start_from": "epicycle"}})
    logs = []
    c._append_log = logs.append
    _pick(c, w, "nuclei")
    assert w.post_small_obj_sb.value() == 100
    assert c._start_from_key() == "epicycle"
    assert c.state_manager.get_run_specks() == 100       # the memory follows
    _pick(c, w, "cyto3")
    assert w.post_small_obj_sb.value() == 10000
    assert c._start_from_key() == "epicycle"
    assert P.S3_LOG_NUCLEI_SPECKS.format(n=100) in logs
    assert P.S3_LOG_MODEL_OVERLAY.format(n=10000) in logs


def test_a_model_pick_under_custom_writes_nothing(w, tmp_path):
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _hand(w, specks=4321)
    _pick(c, w, "nuclei")
    assert w.post_small_obj_sb.value() == 4321
    assert c._start_from_key() == "custom"


def test_the_restores_re_save_of_a_model_writes_no_setting(w, tmp_path):
    """Only a model PICK (it names its mode) re-applies the overlay."""
    c = _open(w, tmp_path, {"default": {"patches": {},
                                        "start_from": "epicycle"}})
    w.lateral_model_path.setText("nuclei")
    c._persist_current_model_selection()
    assert w.post_small_obj_sb.value() == 10000


def test_a_switch_between_a_nuclei_run_and_a_custom_cell_run(w, tmp_path):
    """Neither run's specks leak into the other (both directions)."""
    nuc = {"patches": {}, "start_from": "epicycle",
           "lateral_model_path": "nuclei", "axial_model_path": "nuclei",
           "model_source_mode": "pretrained", "channels": [1, 0]}
    cells = {"patches": {}, "start_from": "custom", "display_name": "cells",
             "lateral_model_path": "cyto3", "axial_model_path": "cyto3",
             "model_source_mode": "pretrained",
             "channels": [2, 1], "specks": 10000}
    c = _open(w, tmp_path, {"default": nuc, "cells": cells})
    # the workspace open restores the model before the Start from choice
    assert c._restore_persisted_model_selection()
    c._restore_start_from()
    assert _boxes(w) == (1, 0, 100)
    c._switch_active_run("cells")
    assert _boxes(w) == (2, 1, 10000)
    c._switch_active_run("default")
    assert _boxes(w) == (1, 0, 100)


def test_the_overlay_table_is_separate_from_the_epicycle_table():
    from napari_epicycle.ui_shell import _recon_presets as R
    assert R.STEP4_PRESET_EPICYCLE["post_small_obj_sb"] == 10000
    assert R.model_overlay("nuclei") == {"post_small_obj_sb": 100}
    assert R.model_overlay(" nuclei ") == {"post_small_obj_sb": 100}
    assert R.model_overlay("cyto3") == {}
    assert R.model_overlay("C:/models/nuclei") == {}
    assert R.model_overlay(None) == {}
    R.model_overlay("nuclei")["post_small_obj_sb"] = 1      # a copy
    assert R.MODEL_PRESET_OVERLAYS["nuclei"]["post_small_obj_sb"] == 100


# ------------- a run with no memory shows the channels its records asked for

def _pred_record(c, name, codes, *, one_channel=False, mtime=None,
                 when=None):
    """A prediction manifest of the ACTIVE run (requested codes)."""
    from napari_epicycle.fusion_reconstruction import _step3_manifests as S
    cache = c._prediction_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    rec = {"manifest_version": 1, "patch_name": name, "axes": "CZYX",
           "seg_config": {"channels": codes}}
    if when is not None:
        rec["predicted_at"] = when
    if one_channel:
        rec["one_channel"] = True
    path = S.prediction_manifest_path(cache, name)
    S.write_json_atomic(path, rec)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _recon_record(c, name, codes, mtime=None, when=None):
    from napari_epicycle.fusion_reconstruction import _paths
    from napari_epicycle.fusion_reconstruction.controller import (
        _active_run_path_id,
    )
    root = Path(_paths.reconstructions_dir(c.working_dir,
                                           _active_run_path_id(c)))
    d = root / Path(name).stem
    d.mkdir(parents=True, exist_ok=True)
    path = d / "manifest.json"
    rec = {"recon_config": {}, "seg_config": {"channels": codes}}
    if when is not None:
        rec["generated_at"] = when
    path.write_text(json.dumps(rec), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_an_older_run_opens_at_the_channels_its_prediction_asked_for(
        w, tmp_path):
    """The owner's ``nuclei`` run, predicted with [0, 1], no memory: it
    opens at 0: gray / 1: red (not the build defaults 2: green / 1: red),
    and nothing is stored until the run is left."""
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (2, 1)
    _pred_record(c, "P1.tif", [0, 1])
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (0, 1)
    assert c.state_manager.get_run_channels() is None
    assert c._start_from_key() == "custom"


def test_the_seed_skips_blank_records_but_keeps_one_channel_ones(
        w, tmp_path):
    """A one-channel patch's record keeps the codes the run ASKED for (it
    ran as gray, but ``seg_config.channels`` is the request): it counts.
    A blank channel to segment (``null``) does not."""
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _pred_record(c, "old.tif", [1, 0], mtime=1_000_000)
    _pred_record(c, "gray.tif", [3, 2], one_channel=True, mtime=2_000_000)
    _pred_record(c, "blank.tif", None, mtime=3_000_000)
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (3, 2)


def test_the_newest_record_is_by_its_own_time_across_both_kinds(
        w, tmp_path):
    """"Newest" is the time each manifest records (``predicted_at`` /
    ``generated_at``), not the file's mtime (a copy rewrites files), and a
    reconstruction newer than every prediction wins."""
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    # the files' mtimes say the opposite of the recorded times
    _pred_record(c, "p1.tif", [2, 1], mtime=9_000_000,
                 when="2026-10-20T10:00:00+00:00")
    _recon_record(c, "p2.tif", [1, 0], mtime=1_000_000,
                  when="2026-10-26T10:00:00+00:00")
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (1, 0)
    _pred_record(c, "p3.tif", [3, 0], mtime=1_000_000,
                 when="2026-10-27T10:00:00+00:00")
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (3, 0)


def test_the_outgoing_seed_stores_the_records_codes_first(w, tmp_path):
    """The run being left stores what its records say (what the restore
    showed), even if the boxes were moved without a write."""
    c = _open(w, tmp_path, {
        "default": dict(A_CUSTOM),
        "b": {"patches": {}, "display_name": "b", "start_from": "custom",
              "channels": [2, 1], "specks": 300}})
    _pred_record(c, "P1.tif", [0, 1])
    c = _open(w, tmp_path)
    c._run_switch_in_flight = True              # moved, nothing written
    w.chan1_combo.setCurrentIndex(3)
    c._run_switch_in_flight = False
    c._seed_outgoing_run_memory()
    assert c.state_manager.get_run_channels() == [0, 1]


def test_the_seed_falls_back_to_the_newest_reconstruction(w, tmp_path):
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _recon_record(c, "a.tif", [3, 0], mtime=1_000_000)
    _recon_record(c, "b.tif", [1, 2], mtime=2_000_000)
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (1, 2)


def test_a_run_with_no_records_keeps_the_boxes(w, tmp_path):
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    w.chan1_combo.setCurrentIndex(3)
    w.chan2_combo.setCurrentIndex(2)
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (3, 2)
    assert c.state_manager.get_run_channels() is None


def test_a_stored_memory_wins_over_the_records(w, tmp_path):
    c = _open(w, tmp_path, {"default": {
        "patches": {}, "start_from": "custom",
        "channels": [3, 0], "specks": 300}})
    _pred_record(c, "P1.tif", [0, 1])
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (3, 0)


def test_switching_into_an_older_run_shows_its_records_and_stores_on_leave(
        w, tmp_path):
    """Run switch INTO a run with no memory: its newest prediction's codes
    are shown, nothing is written during the switch; the first switch
    away stores them (the outgoing seed)."""
    c = _open(w, tmp_path, {
        "default": {"patches": {}, "start_from": "custom",
                    "channels": [2, 1], "specks": 300},
        "nuclei": {"patches": {}, "display_name": "Nuclei",
                   "start_from": "custom"}})
    c._switch_active_run("nuclei")
    assert c.state_manager.get_active_run_id() == "nuclei"
    _pred_record(c, "P1.tif", [0, 1])
    c._switch_active_run("default")
    assert _boxes(w)[:2] == (2, 1)
    # the first visit seeded nothing: the run had no record then, so the
    # outgoing seed stored the boxes it showed ([2, 1]); clear it to
    # replay the owner's case -- a run whose memory was never written
    c.state_manager.flush()
    runs = _read_runs(tmp_path)
    runs["nuclei"].pop("channels", None)
    runs["nuclei"].pop("specks", None)
    _write_state(tmp_path, runs)
    c = _open(w, tmp_path)
    c._switch_active_run("nuclei")
    assert _boxes(w)[:2] == (0, 1)
    assert c.state_manager.get_run_channels() is None
    c._switch_active_run("default")
    assert _boxes(w)[:2] == (2, 1)
    c.state_manager.flush()
    assert _read_runs(tmp_path)["nuclei"]["channels"] == [0, 1]
    c._switch_active_run("nuclei")
    assert _boxes(w)[:2] == (0, 1)


def test_the_first_hand_change_after_the_seed_is_stored(w, tmp_path):
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _pred_record(c, "P1.tif", [0, 1])
    c = _open(w, tmp_path)
    assert c.state_manager.get_run_channels() is None
    _hand(w, c2=2)
    assert c.state_manager.get_run_channels() == [0, 2]


# ----------------------------------- the foot and the run combo on a switch

def test_a_new_run_starts_with_an_empty_bar_and_its_own_line(w, tmp_path,
                                                             monkeypatch):
    """The previous run's Predict left the bar at 100 % and its done line:
    the new run (activated) shows neither."""
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    w.lateral_model_path.setText("cyto3")
    w.axial_model_path.setText("cyto3")
    bar = w.pipeline_progress
    bar.setRange(0, 40)
    bar.setValue(40)
    done = "Segmentation done: 1/1 patch [0:41]"
    c._pr_status(done)
    _new_run(c, monkeypatch, "nb", _pretrained("nuclei"))
    assert c.state_manager.get_active_run_id() == "nb"
    assert bar.value() < bar.minimum(), bar.value()
    assert c._pr_job_line != done
    assert w.pipeline_status_label.text() != done
    # and a plain switch back does the same
    bar.setRange(0, 40)
    bar.setValue(40)
    c._pr_status(done)
    c._switch_active_run("default")
    assert bar.value() < bar.minimum()
    assert w.pipeline_status_label.text() != done


def _combo_text(w, slug):
    combo = w.run_picker_combo
    return combo.itemText(combo.findData(slug))


def test_the_run_combo_counts_the_active_run_from_memory(w, tmp_path,
                                                         monkeypatch):
    """A run just created reads its patches from memory, not from the
    state on disk (saved a moment later): never "(0 patches)" beside a
    full list; after a Predict or a Reconstruct the count follows."""
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _new_run(c, monkeypatch, "nb", _pretrained("nuclei"))
    c.state_manager.register_patches(["a.tif", "b.tif", "c.tif"])
    c._refresh_run_picker()
    assert "(3 patches)" in _combo_text(w, "nb")
    c.state_manager.set_patch_status("d.tif", "predicted")
    c._pipeline_patch_count, c._pipeline_errors = 1, 0
    c._pipeline_start_time = 0.0
    c._on_serial_pred_complete()
    assert "(4 patches)" in _combo_text(w, "nb")
    c.state_manager.set_patch_status("e.tif", "reconstructed")
    c._on_serial_recon_complete()
    assert "(5 patches)" in _combo_text(w, "nb")
    # Forget: the list and the combo agree again
    c.state_manager.forget_patch("a.tif")
    c._refresh_patch_state_list()
    assert "(4 patches)" in _combo_text(w, "nb")
