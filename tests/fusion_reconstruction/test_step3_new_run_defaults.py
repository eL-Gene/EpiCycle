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
* A run with no channels remembered and no prediction or reconstruction
  record (the workspace's first run, or an older run that never
  predicted) is seeded from its CURRENT model by a new run's rule: the
  model's trained codes, else a blank "Channel to segment".  Shown, never
  stored, at the open, a switch and a model pick -- and an untouched seed
  is not stored when the run is left either, so the run keeps following
  its model; the first hand change (or a bundle's channels synced at a
  pick) stores it.  A record or a memory always wins.
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


def _models(w, lateral, axial=None):
    """The model fields as the open path leaves them."""
    w.lateral_model_path.setText(str(lateral))
    w.axial_model_path.setText(str(lateral if axial is None else axial))


def _logs(c):
    logs = []
    c._append_log = logs.append
    return logs


def _seed_line(kind, run="Default", **kw):
    """The seed line *kind* (``S3_LOG_SEED_<kind>``) as the log shows it."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    head = P.S3_LOG_SEED_HEAD.format(run=run)
    return getattr(P, f"S3_LOG_SEED_{kind}").format(head=head, **kw)


#: the mixed line when the lateral view's model records its channels
LAT_ONLY = {"with_view": "lateral", "without_view": "axial"}


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
    """A run with no memory and no record (made before runs remembered
    their channels) shows its model's seed -- a built-in's blank box, never
    the build defaults.  Leaving it stores its specks but NOT the untouched
    seed; a switch back seeds it again, never the other run's channels."""
    run = {"patches": {}, "lateral_model_path": "cyto3",
           "axial_model_path": "cyto3", "model_source_mode": "pretrained"}
    if stored:
        run["start_from"] = stored
    c = _open(w, tmp_path, {"default": run})
    c._restore_persisted_model_selection()
    c._restore_start_from()
    before = _boxes(w)
    assert before[:2] == (-1, 0)                     # cyto3 records none
    assert c.state_manager.get_run_channels() is None
    _new_run(c, monkeypatch, "nuc", _pretrained("nuclei"))
    w.chan1_combo.setCurrentIndex(1)                 # the pick: 1: red
    c._switch_active_run("default")
    assert _boxes(w) == before
    assert c.state_manager.get_run_channels() is None
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
    opens at 0: gray / 1: red -- not the build defaults 2: green / 1: red,
    and not its model's seed (the record wins) -- and nothing is stored
    until the run is left."""
    model = _trained_model(tmp_path / "models" / "m3", [3, 0])
    _models(w, model)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (3, 0)                   # no record: the seed
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


def test_a_run_with_no_records_takes_its_model_not_the_boxes(w, tmp_path):
    """No memory, no record: whatever the boxes hold (here moved to 3 / 2
    without a write) gives way to the model's codes on the next open."""
    model = _trained_model(tmp_path / "models" / "m1", [1, 0])
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    c._run_switch_in_flight = True                   # moved, nothing written
    w.chan1_combo.setCurrentIndex(3)
    w.chan2_combo.setCurrentIndex(2)
    c._run_switch_in_flight = False
    assert c.state_manager.get_run_channels() is None
    _models(w, model)
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (1, 0)
    assert c.state_manager.get_run_channels() is None
    # no model at all: blank, never the moved 3 / 2
    w.chan1_combo.setCurrentIndex(3)
    w.chan2_combo.setCurrentIndex(2)
    _models(w, "")
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (-1, 0)


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


# ------------- a run with nothing remembered or recorded: seeded from its model

def _step2_model(ws: Path, run: str, bucket: str, codes, mtime=None) -> Path:
    """A Step 2 round model where the workspace scan finds it, with its
    training config beside it."""
    folder = (ws / "artifacts" / "step2" / "runs" / run / bucket / "models"
              / "round_01")
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / f"{run}_{bucket}_round_01"
    weights.write_bytes(b"weights")
    (folder / f"{weights.name}_training_config.json").write_text(
        json.dumps({"hyperparameters": {"channels": list(codes)}}),
        encoding="utf-8")
    if mtime is not None:
        os.utime(weights, (mtime, mtime))
    return weights


def test_a_fresh_default_run_takes_its_models_channels(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    model = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, model)
    logs = _logs(w.controller)
    c = _open(w, tmp_path, {"default": {"patches": {},
                                        "start_from": "epicycle"}})
    assert _boxes(w)[:2] == (1, 0)
    assert c.state_manager.get_run_channels() is None       # shown only
    assert c._active_run_display_name() == "Default"
    assert _seed_line("CHANNELS", seg=P.S3_CHAN1_ITEMS[1],
                      helper=P.S3_CHAN2_ITEMS[0]) in logs
    run_memory = P.S3_LOG_RUN_MEMORY.split("{")[0]
    assert not [m for m in logs if m.startswith(run_memory)], logs
    c.state_manager.flush()
    assert "channels" not in _read_runs(tmp_path)["default"]


def test_a_fresh_default_run_on_a_built_in_is_blank(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    _models(w, "cyto3")
    logs = _logs(w.controller)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (-1, 0)
    assert w.chan1_combo.placeholderText() == P.S3_CHAN1_PLACEHOLDER
    assert c.state_manager.get_run_channels() is None
    assert _seed_line("PICK") in logs


def test_a_fresh_default_run_with_no_model_is_blank(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    _models(w, "")
    logs = _logs(w.controller)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (-1, 0)
    assert c.state_manager.get_run_channels() is None
    assert _seed_line("NO_MODEL") in logs
    # one view without a model says the same
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1, "")
    logs.clear()
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (-1, 0)
    assert _seed_line("NO_MODEL") in logs


def test_two_views_trained_differently_say_so(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    m2 = _trained_model(tmp_path / "models" / "m2", [2, 1])
    _models(w, m1, m2)
    logs = _logs(w.controller)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (-1, 0)
    assert _seed_line("DISAGREE") in logs
    pick = _seed_line("PICK")
    assert pick not in logs
    assert c.state_manager.get_run_channels() is None


def test_only_remembered_specks_get_their_own_line(w, tmp_path):
    """A run that remembers its specks but no channels: the seed line says
    where the channels come from, and the specks get their own line -- not
    the "This run's channels" line, which would contradict it."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    logs = _logs(w.controller)
    _open(w, tmp_path, {"default": {"patches": {}, "start_from": "custom",
                                    "specks": 250}})
    assert _boxes(w) == (1, 0, 250)
    assert P.S3_LOG_RUN_SPECKS.format(n=250) in logs
    run_memory = P.S3_LOG_RUN_MEMORY.split("{")[0]
    assert not [m for m in logs if m.startswith(run_memory)], logs


def _seed_lines(logs):
    """Every seed line in *logs*, found by the words after its head."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    words = [t.split("{head}")[1].split("{")[0] for t in (
        P.S3_LOG_SEED_CHANNELS, P.S3_LOG_SEED_PICK, P.S3_LOG_SEED_DISAGREE,
        P.S3_LOG_SEED_MIXED, P.S3_LOG_SEED_NO_MODEL)]
    head = P.S3_LOG_SEED_HEAD.split("{run}")[0]
    return [m for m in logs
            if m.startswith(head) and any(wd in m for wd in words)]


def test_no_seed_runs_when_the_run_remembers(w, tmp_path):
    """A memory: the seed is not run at all (its line absent), on the open
    and on a pick."""
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    logs = _logs(w.controller)
    c = _open(w, tmp_path, {"default": {"patches": {}, "start_from": "custom",
                                        "channels": [3, 2]}})
    assert _boxes(w)[:2] == (3, 2)
    c._persist_current_model_selection(mode="import")
    assert _seed_lines(logs) == []


def test_no_seed_runs_when_the_run_has_a_record(w, tmp_path):
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _pred_record(c, "P1.tif", [2, 1])
    logs = _logs(c)
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (2, 1)
    c._persist_current_model_selection(mode="import")
    assert _seed_lines(logs) == []


def test_a_named_run_with_nothing_known_is_seeded_too(w, tmp_path):
    """Not only the default run: any run with no memory and no record (a
    named run made before the runs remembered, never predicted) is seeded
    from the model the switch restores for it."""
    model = _trained_model(tmp_path / "models" / "m2", [2, 1])
    c = _open(w, tmp_path, {
        "default": {"patches": {}, "start_from": "custom",
                    "channels": [3, 0], "specks": 300},
        "old": {"patches": {}, "display_name": "old",
                "start_from": "custom",
                "lateral_model_path": str(model),
                "axial_model_path": str(model),
                "model_source_mode": "import"}})
    assert _boxes(w)[:2] == (3, 0)
    c._switch_active_run("old")
    assert w.lateral_model_path.text() == str(model)
    assert _boxes(w)[:2] == (2, 1)
    assert c.state_manager.get_run_channels() is None
    c._switch_active_run("default")
    assert _boxes(w)[:2] == (3, 0)
    c.state_manager.flush()
    assert "channels" not in _read_runs(tmp_path)["old"]


def test_the_open_path_discovers_then_seeds(w, tmp_path):
    """The real open path: the model is found by the workspace scan (not
    set by hand), and the boxes follow it."""
    _step2_model(tmp_path, "r1", "lateral", [1, 0])
    _step2_model(tmp_path, "r1", "axial", [1, 0])
    _write_state(tmp_path, {"default": {"patches": {},
                                        "start_from": "epicycle"}})
    c = w.controller
    w.chan1_combo.setCurrentIndex(2)                 # the build defaults
    w.chan2_combo.setCurrentIndex(1)
    c._adopt_workspace_quiet(str(tmp_path))
    assert w.lateral_model_path.text().endswith("r1_lateral_round_01")
    assert w.axial_model_path.text().endswith("r1_axial_round_01")
    assert _boxes(w)[:2] == (1, 0)
    assert c.state_manager.get_run_channels() is None


def test_two_step2_runs_with_different_channels_leave_the_box_blank(
        w, tmp_path):
    """Auto-discovery takes the newest model per view: here the lateral of
    one Step 2 run (trained [1, 0]) and the axial of another (trained
    [2, 1]).  The two views disagree: blank, never a mixed guess."""
    _step2_model(tmp_path, "a", "lateral", [1, 0], mtime=2_000_000)
    _step2_model(tmp_path, "a", "axial", [1, 0], mtime=1_000_000)
    _step2_model(tmp_path, "b", "lateral", [2, 1], mtime=1_000_000)
    _step2_model(tmp_path, "b", "axial", [2, 1], mtime=2_000_000)
    _write_state(tmp_path, {"default": dict(A_CUSTOM)})
    c = w.controller
    c._adopt_workspace_quiet(str(tmp_path))
    assert "a_lateral" in Path(w.lateral_model_path.text()).name
    assert "b_axial" in Path(w.axial_model_path.text()).name
    assert _boxes(w)[:2] == (-1, 0)


def test_a_second_workspace_in_one_session_gets_its_own_seed(
        w, tmp_path):
    """The widgets live across workspaces: the second workspace's run
    never shows the first one's channels."""
    ws1, ws2, ws3 = (tmp_path / n for n in ("ws1", "ws2", "ws3"))
    for ws in (ws1, ws2, ws3):
        ws.mkdir()
        _write_state(ws, {"default": dict(A_CUSTOM)})
    _step2_model(ws1, "r", "lateral", [1, 0])
    _step2_model(ws1, "r", "axial", [1, 0])
    _step2_model(ws3, "r", "lateral", [3, 2])
    _step2_model(ws3, "r", "axial", [3, 2])
    c = w.controller
    c._adopt_workspace_quiet(str(ws1))
    assert _boxes(w)[:2] == (1, 0)
    c._adopt_workspace_quiet(str(ws2))               # no model at all
    assert w.lateral_model_path.text() == ""
    assert _boxes(w)[:2] == (-1, 0)
    c._adopt_workspace_quiet(str(ws3))
    assert _boxes(w)[:2] == (3, 2)
    c.state_manager.flush()
    for ws in (ws1, ws2, ws3):
        assert "channels" not in _read_runs(ws)["default"], ws


def test_a_model_pick_re_seeds_while_nothing_is_known(w, tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    m3 = _trained_model(tmp_path / "models" / "m3", [3, 0])
    _models(w, m1)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (1, 0)
    logs = _logs(c)
    _models(w, m3)
    c._persist_current_model_selection(mode="import")      # Browse
    assert _boxes(w)[:2] == (3, 0)
    assert c.state_manager.get_run_channels() is None       # shown only
    assert c._start_from_key() == "custom"                  # untouched
    assert _seed_line("CHANNELS", seg=P.S3_CHAN1_ITEMS[3],
                      helper=P.S3_CHAN2_ITEMS[0]) in logs
    # a built-in picked: blank
    _models(w, "cyto3")
    c._persist_current_model_selection(mode="pretrained")
    assert _boxes(w)[:2] == (-1, 0)
    assert c.state_manager.get_run_channels() is None
    # the restore's re-save (no mode) is not a pick
    _models(w, m1)
    c._persist_current_model_selection()
    assert _boxes(w)[:2] == (-1, 0)
    # after a hand change the run remembers: a pick changes nothing
    _hand(w, c1=2, c2=1)
    assert c.state_manager.get_run_channels() == [2, 1]
    _models(w, m3)
    c._persist_current_model_selection(mode="import")
    assert _boxes(w)[:2] == (2, 1)
    assert c.state_manager.get_run_channels() == [2, 1]


def test_a_model_pick_after_a_record_re_seeds_nothing(w, tmp_path):
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    _pred_record(c, "P1.tif", [2, 1])
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (2, 1)
    _models(w, m1)
    c._persist_current_model_selection(mode="import")
    assert _boxes(w)[:2] == (2, 1)
    assert c.state_manager.get_run_channels() is None


def test_a_stored_record_wins_over_the_seed(w, tmp_path):
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (1, 0)
    _recon_record(c, "P1.tif", [2, 1])
    c = _open(w, tmp_path)
    assert _boxes(w)[:2] == (2, 1)


def test_a_stored_memory_wins_over_the_seed(w, tmp_path):
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    _open(w, tmp_path, {"default": {"patches": {}, "start_from": "custom",
                                    "channels": [3, 2], "specks": 300}})
    assert _boxes(w) == (3, 2, 300)


def _two_runs(w, tmp_path):
    return _open(w, tmp_path, {
        "default": dict(A_CUSTOM),
        "b": {"patches": {}, "display_name": "b", "start_from": "custom",
              "channels": [3, 2], "specks": 500}})


def test_the_first_switch_away_keeps_an_untouched_seed_unstored(
        w, tmp_path):
    """An untouched seed is not the run's memory: leaving stores the specks
    but no channels, so the run keeps following its model."""
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    c = _two_runs(w, tmp_path)
    assert _boxes(w)[:2] == (1, 0)
    c._switch_active_run("b")
    assert _boxes(w)[:2] == (3, 2)
    c.state_manager.flush()
    got = _read_runs(tmp_path)["default"]
    assert "channels" not in got
    assert got["specks"] == 10000


def test_after_a_switch_away_a_pick_still_re_seeds(w, tmp_path):
    """The bug an eager store caused: a [1, 0] seed frozen on the switch
    away, then a model trained [3, 0] browsed -- Predict would record
    [1, 0] for it.  The boxes follow the picked model."""
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    m3 = _trained_model(tmp_path / "models" / "m3", [3, 0])
    _models(w, m1)
    c = _two_runs(w, tmp_path)
    c._switch_active_run("b")
    c._switch_active_run("default")
    _models(w, m3)
    c._persist_current_model_selection(mode="import")
    assert _boxes(w)[:2] == (3, 0)
    assert c.state_manager.get_run_channels() is None
    assert c._collect_seg_config().channels == [3, 0]


def test_a_hand_edit_then_a_switch_away_is_stored(w, tmp_path):
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    c = _two_runs(w, tmp_path)
    _hand(w, c2=2)
    c._switch_active_run("b")
    c.state_manager.flush()
    assert _read_runs(tmp_path)["default"]["channels"] == [1, 2]
    c._switch_active_run("default")
    assert _boxes(w)[:2] == (1, 2)


def _bundle(folder: Path, lateral_codes, axial_codes) -> Path:
    """A refined bundle: both weights, a training config per view (or none
    for a view given None)."""
    folder.mkdir(parents=True, exist_ok=True)
    for bucket, codes in (("lateral", lateral_codes),
                          ("axial", axial_codes)):
        (folder / bucket).write_bytes(b"weights")
        if codes is not None:
            (folder / f"{bucket}_training_config.json").write_text(
                json.dumps({"hyperparameters": {"channels": list(codes)}}),
                encoding="utf-8")
    return folder


def _bundle_pick(c, w, bundle: Path):
    """The registry's Use / Use for re-prediction: the paths, the bundle
    sync, then the pick."""
    _models(w, bundle / "lateral", bundle / "axial")
    c._try_sync_prediction_channels_from_bundle(str(bundle))
    c._persist_current_model_selection(mode="import")


def test_a_bundle_pick_with_the_codes_already_shown_says_one_thing(
        w, tmp_path, monkeypatch):
    """The boxes already show the bundle's [1, 0] (the run's seed), so
    the sync changes no index.  The sync is the pick: it is remembered once,
    and no seed runs after it (no line saying nothing is remembered)."""
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (1, 0)
    stores = []
    real = c.state_manager.set_run_channels
    monkeypatch.setattr(c.state_manager, "set_run_channels",
                        lambda v: stores.append(list(v)) or real(v))
    logs = _logs(c)
    _bundle_pick(c, w, _bundle(tmp_path / "bundle", [1, 0], [1, 0]))
    assert _boxes(w)[:2] == (1, 0)
    assert stores == [[1, 0]]
    assert c.state_manager.get_run_channels() == [1, 0]
    assert any(m.startswith("[CHANNEL-SYNC] applied") for m in logs), logs
    assert _seed_lines(logs) == []


def test_a_bundle_without_a_lateral_config_keeps_the_synced_codes(
        w, tmp_path):
    """The bundle records its channels only beside the axial weights.
    The sync applies [1, 0] (it falls back to the axial config); the pick
    must not then blank the box through the per-view seed."""
    _models(w, "cyto3")
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (-1, 0)
    logs = _logs(c)
    _bundle_pick(c, w, _bundle(tmp_path / "bundle", None, [1, 0]))
    assert _boxes(w)[:2] == (1, 0)
    assert c.state_manager.get_run_channels() == [1, 0]
    assert _seed_lines(logs) == []


def test_predict_after_the_seed_records_it(w, tmp_path, monkeypatch):
    """The seeded codes are what Predict hands Cellpose and records in the
    prediction manifest, and what Reconstruct copies into manifest.json."""
    import numpy as np
    import tifffile

    import packages.epicycle_reconstruction.logic as logic
    from napari_epicycle.fusion_reconstruction import _step3_manifests as S

    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (1, 0)
    folder = tmp_path / "patches"
    folder.mkdir()
    stack = folder / "P1.tif"
    tifffile.imwrite(str(stack), np.zeros((2, 6, 8, 8), np.uint8),
                     metadata={"axes": "CZYX"})
    handed = []

    def fake_segment(image, lat, ax, seg_config, axes="ZCYX",
                     progress_cb=None):
        handed.append(list(seg_config.channels))
        f = np.ones((2, 3, 4, 4), np.float32)
        p = np.ones((3, 4, 4), np.float32)
        return {v: (f, p, None) for v in ("xy", "xz", "yz")}

    monkeypatch.setattr(logic, "segment_views", fake_segment)
    c._predict_single_patch(
        patch_name="P1.tif", stack_path=str(stack), lateral_path=str(m1),
        axial_path=str(m1), seg_config=c._collect_seg_config(),
        axes="CZYX")
    assert handed == [[1, 0]]
    pm = json.loads(S.prediction_manifest_path(
        c._prediction_cache_dir(), "P1.tif").read_text(encoding="utf-8"))
    assert pm["seg_config"]["channels"] == [1, 0]
    # Reconstruct: run_real faked; its arguments build manifest.json
    seen = {}

    def run_real(**kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(c.reconstruction_service, "run_real", run_real)
    _hand(w, c1=3, c2=2)              # today's boxes are not the record
    c._reconstruct_single_patch(
        patch_name="P1.tif", stack_path=str(stack),
        working_dir=c.working_dir, lateral_path=str(m1),
        axial_path=str(m1), seg_config=c._collect_seg_config(),
        recon_config=None, cell_width_um=8.0)
    manifest = logic.build_real_manifest(
        run_id="r", source_stack_path=str(stack), labels_path="l.tif",
        labels_shape=[6, 8, 8], num_cells_raw=1, num_cells_final=1,
        axes=seen["axes"], seg_config=seen["seg_config"],
        recon_config=logic.ReconstructionConfig(),
        extra_manifest=seen["extra_manifest"])
    manifest = json.loads(json.dumps(manifest))
    assert manifest["seg_config"]["channels"] == [1, 0]
    assert manifest["seg_config_source"] == "prediction"


# --------------------------------------------- the seed flag and its words

def test_a_run_shown_from_its_record_does_not_inherit_the_seed_flag(
        w, tmp_path):
    """Run A shows its model's seed (untouched); run B has no memory but a
    prediction [2, 1], so it shows 2 / 1.  B's record goes while B is
    active (its manifest removed); leaving B must store the 2 / 1 it
    showed -- A's "untouched seed" must not follow B."""
    c = _open(w, tmp_path, {
        "default": dict(A_CUSTOM),
        "b": {"patches": {}, "display_name": "b", "start_from": "custom"}})
    c._switch_active_run("b")
    rec = _pred_record(c, "P1.tif", [2, 1])
    c._switch_active_run("default")
    c.state_manager.flush()
    runs = _read_runs(tmp_path)
    runs["b"].pop("channels", None)
    runs["default"].pop("channels", None)
    _write_state(tmp_path, runs)
    c = _open(w, tmp_path)                   # A: the seed, untouched
    assert c._channels_seed_shown is True
    c._switch_active_run("b")
    assert _boxes(w)[:2] == (2, 1)
    assert c.state_manager.get_run_channels() is None
    os.remove(rec)                           # Forget / Reset: the record goes
    c._switch_active_run("default")
    c.state_manager.flush()
    assert _read_runs(tmp_path)["b"]["channels"] == [2, 1]


def test_returning_to_an_untouched_run_says_only_its_channels_are_unsaved(
        w, tmp_path):
    """Leaving an untouched run saves its specks, not its channels.  On the
    way back the seed line says no channels are saved -- never "nothing
    saved" (its model, Start from and specks are) -- and the specks get one
    line of their own."""
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1)
    c = _open(w, tmp_path, {
        "default": dict(A_CUSTOM),
        "b": {"patches": {}, "display_name": "b", "start_from": "custom",
              "channels": [3, 2], "specks": 500,
              "lateral_model_path": str(m1), "axial_model_path": str(m1),
              "model_source_mode": "import"}})
    c._switch_active_run("b")
    assert c.state_manager.get_run_specks() == 500
    logs = _logs(c)
    c._switch_active_run("default")
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    assert c.state_manager.get_run_channels() is None
    assert c.state_manager.get_run_specks() == 10000
    seeds = _seed_lines(logs)
    assert len(seeds) == 1, logs
    assert seeds[0].startswith(P.S3_LOG_SEED_HEAD.format(run="Default"))
    assert not [m for m in logs if "nothing saved" in m], logs
    specks = [m for m in logs if "Remove specks smaller than" in m]
    assert specks == [P.S3_LOG_RUN_SPECKS.format(n=10000)], logs


def test_the_seed_and_specks_lines_read_as_one_sentence_each():
    """One colon in the channels clause (the item words carry their own),
    no double space, no colon before the specks number, and the specks
    line worded like the other specks lines."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    line = _seed_line("CHANNELS", seg="1: red", helper="0: none")
    assert "  " not in line
    assert "trained on:" not in line
    assert "— channel to segment 1: red, helper 0: none." in line
    assert "smaller than:" not in P.S3_LOG_RUN_SPECKS
    assert "Remove specks smaller than {n} voxels." in P.S3_LOG_RUN_SPECKS
    # a separator like its siblings ("This run’s channels: ...")
    assert P.S3_LOG_RUN_SPECKS.startswith(
        "[SETTINGS] This run\u2019s settings: Remove specks")
    assert "smaller than {n} voxels" in P.S3_LOG_NUCLEI_SPECKS
    for kind in ("CHANNELS", "PICK", "DISAGREE", "MIXED", "NO_MODEL"):
        text = getattr(P, f"S3_LOG_SEED_{kind}")
        assert text.startswith("{head}"), kind
        assert "  " not in text, kind
    head = P.S3_LOG_SEED_HEAD
    assert "remembered" not in head and "recorded" not in head
    # the run's model and Start from are saved: never "nothing saved"
    assert "nothing saved" not in head
    assert head.format(run="b") == (
        "[SETTINGS] Step 3 run b: no channels saved for this run yet, and "
        "not predicted yet")
    assert not hasattr(P, "S3_LOG_SEED_HEAD_OTHERS")


def test_one_view_trained_and_one_not_says_so(w, tmp_path):
    """A trained model on one view and a built-in on the other: not "its
    model does not record its channels" (one of them does)."""
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, m1, "cyto3")
    logs = _logs(w.controller)
    _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _boxes(w)[:2] == (-1, 0)
    assert _seed_line("MIXED", **LAT_ONLY) in logs
    assert _seed_line("PICK") not in logs
    assert _seed_line("DISAGREE") not in logs


def test_the_mixed_line_names_the_view_without_channels(w, tmp_path):
    """The trained model on the AXIAL view: the line names the lateral view
    as the one to pick a trained model for -- never "the other view"."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    _models(w, "cyto3", m1)
    logs = _logs(w.controller)
    _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    line = _seed_line("MIXED", with_view="axial", without_view="lateral")
    assert line in logs, logs
    assert line.endswith(
        "its axial model records its channels but its lateral model does "
        "not; pick a trained model for the lateral view, or pick the "
        "channel to segment.")
    assert _seed_line("MIXED", **LAT_ONLY) not in logs
    assert "the other view" not in P.S3_LOG_SEED_MIXED


def test_a_malformed_config_is_not_trained_differently(w, tmp_path):
    """Codes that are not two Cellpose codes ([5, 1]) are no record: with a
    trained view beside it the line is the mixed one; with two such views,
    the model records nothing usable."""
    m1 = _trained_model(tmp_path / "models" / "m1", [1, 0])
    bad = _trained_model(tmp_path / "models" / "bad", [5, 1])
    bad2 = _trained_model(tmp_path / "models" / "bad2", [7, 7])
    _models(w, m1, bad)
    logs = _logs(w.controller)
    _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    assert _seed_line("MIXED", **LAT_ONLY) in logs
    assert _seed_line("DISAGREE") not in logs
    _models(w, bad, bad2)
    logs.clear()
    _open(w, tmp_path)
    assert _seed_line("PICK") in logs
    assert _seed_line("DISAGREE") not in logs


def test_the_run_name_is_never_another_workspaces(w, tmp_path):
    """The picker's cached summaries name runs by slug; a summary listed
    for another workspace must not name this one's run."""
    c = _open(w, tmp_path, {"default": dict(A_CUSTOM)})
    c._run_picker_summaries = {"default": {"display_name": "Old Name"}}
    c._run_picker_summaries_ws = str(tmp_path / "elsewhere")
    assert c._active_run_display_name() == "Default"
    c._run_picker_summaries_ws = c.working_dir
    assert c._active_run_display_name() == "Old Name"


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
