"""Predict's inputs and what it records (F25; audit A1-4, A1-10).

* Rescale OFF used to reach Cellpose as ``False`` = a resize factor of 0,
  and with no diameter cv2 asserted ``inv_scale_x > 0``: Predict crashed.
  Step 4's call site now passes ``None`` (the model's own trained cell
  size) and records the effective factor.
* The ``predicted_with`` snapshot carries the model fingerprints, the
  resolved axes and how, every SegmentationConfig field, the effective
  factor; Reconstruct copies it into ``made_with`` and the run manifest.
* A Start from value saved as a float int (``25.0``) lands in a QSpinBox;
  one value that will not apply is logged and skipped, never aborting.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump, tick


class _FakeCellpose:
    """Cellpose 3.1.1.2's eval, as far as ``rescale`` goes: a factor; with
    no diameter a factor of 0 makes cv2 assert (the F25 crash)."""

    def __init__(self):
        self.seen = []

    def eval(self, img, *, channels, do_3D, normalize, rescale,
             compute_masks, diameter, cellprob_threshold, flow_threshold):
        self.seen.append(rescale)
        if diameter is None and rescale is not None and float(rescale) <= 0:
            raise AssertionError("inv_scale_x > 0")      # what cv2 says
        h, w = img.shape[-2:]
        flows = np.zeros((2, h, w), np.float32)
        return (np.zeros((h, w), np.int32), (None, flows,
                                              np.zeros((h, w), np.float32)),
                None)


def _ready(w, qapp):
    c = w.controller
    c._on_model_use({"_kind": "pretrained", "name": "cyto3"})
    w.set_channel_choice(w.chan1_combo, 0)
    w.set_channel_choice(w.chan2_combo, None)
    pump(qapp)
    return c


@pytest.fixture
def ws(tmp_path):
    return make_workspace(tmp_path, patches=("P1.tif",))


def test_rescale_off_does_not_crash(qapp, ws):
    from packages.epicycle_reconstruction import logic
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        w.rescale_cb.setChecked(False)
        cfg = c._collect_seg_config({}, None)
        assert cfg.rescale is None
        model = _FakeCellpose()
        image = np.zeros((2, 2, 8, 8), np.float32)          # Z C Y X
        logic._segment_view(image, model, "xy", cfg, None)
        assert model.seen == [None, None]
        # the old call really crashed: False is a factor of 0
        cfg.rescale = False
        with pytest.raises(AssertionError, match="inv_scale_x"):
            logic._segment_view(image, _FakeCellpose(), "xy", cfg, None)
        w.rescale_cb.setChecked(True)
        assert c._collect_seg_config({}, None).rescale is True   # today's
    finally:
        close(w)


def test_effective_rescale_factor_recorded(qapp, ws, monkeypatch):
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        seen = []
        monkeypatch.setattr(c, "_launch_prediction_worker", seen.extend)
        tick(w, qapp, "P1.tif")
        for checked, factor in ((False, None), (True, 1.0)):
            seen.clear()
            c._predict_running = False       # the launch was faked
            c._running = {}
            w.rescale_cb.setChecked(checked)
            c._refresh_segment_gates()
            pump(qapp)
            w.predict_btn.click()
            pump(qapp)
            assert len(seen) == 1
            cfg = seen[0]
            assert cfg["cache_meta"]["rescale_factor"] == factor
            assert cfg["model_snapshot"]["seg"]["rescale_factor"] == factor
            assert cfg["cache_meta"]["model"] is cfg["model_snapshot"]
    finally:
        close(w)


def test_the_snapshot_records_every_input(qapp, ws, monkeypatch):
    """V2.6 : model fingerprints, axes + how, shape, dtype, the
    source's fingerprint, every SegmentationConfig and recon field; the
    worker adds the versions."""
    from dataclasses import fields

    from packages.epicycle_reconstruction.config import ReconstructionConfig
    from packages.epicycle_training.config import SegmentationConfig
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        seen = []
        monkeypatch.setattr(c, "_launch_prediction_worker", seen.extend)
        tick(w, qapp, "P1.tif")
        w.predict_btn.click()
        pump(qapp)
        snap = seen[0]["model_snapshot"]
        assert snap["kind"] == "pretrained"            # the chips' keys
        assert "model_fingerprints" in snap
        inp = snap["input"]
        assert inp["axes"] == "CZYX"
        assert inp["axes_how"] in ("metadata", "heuristic", "fallback",
                                   "user")
        assert inp["shape"] and inp["dtype"] == "uint8"
        assert inp["source_fingerprint"]
        for f in fields(SegmentationConfig):
            assert f.name in snap["seg"], f.name
        for f in fields(ReconstructionConfig):
            assert f.name in snap["recon"], f.name
        json.dumps(snap)                                # JSON-safe
        M.finish_run_snapshot(snap, {})
        assert set(snap["versions"]) == {"cellpose", "torch"}
    finally:
        close(w)


def test_model_fingerprints_and_sha1(tmp_path):
    lat = tmp_path / "lat"
    lat.write_bytes(b"weights-a")
    snap = {"lateral": str(lat), "axial": str(lat)}
    fps = M.model_fingerprints(snap)
    assert fps["lateral"]["size"] == len(b"weights-a")
    assert "mtime_ns" in fps["lateral"]
    full = M.finish_run_snapshot(dict(snap, model_fingerprints=fps), {})
    import hashlib
    assert full["model_fingerprints"]["lateral"]["sha1"] == \
        hashlib.sha1(b"weights-a").hexdigest()


def test_the_manifest_gets_made_with(qapp, ws, tmp_path):
    w = panel(qapp, ws)
    try:
        man = tmp_path / "manifest.json"
        man.write_text(json.dumps({"schema_version": "2.0"}),
                       encoding="utf-8")
        made = {"kind": "pretrained", "recon": {"n_cpu": 6}}
        w.controller._write_manifest_made_with(str(man), made)
        data = json.loads(man.read_text(encoding="utf-8"))
        assert data["made_with"] == made
        assert data["schema_version"] == "2.0"
    finally:
        close(w)


def test_start_from_takes_float_ints_and_skips_a_bad_value(qapp, ws):
    """A1-4: ``25.0`` reaches a QSpinBox as 25; a value that cannot apply
    is logged and skipped; the rest still land."""
    w = panel(qapp, ws)
    try:
        c = w.controller
        values = {"gd_iters_sb": 25.0, "n_cpu_sb": "not a number",
                  "gd_decay_sb": 0.002, "gd_renorm_cb": False}
        c._preset_values = lambda key: values           # noqa: E731
        c._apply_start_from("custom", save=False)
        assert w.gd_iters_sb.value() == 25
        assert w.gd_decay_sb.value() == pytest.approx(0.002)
        assert w.gd_renorm_cb.isChecked() is False
        assert T.LOG_PRESET_SKIPPED.format(name="n_cpu_sb") in \
            w.log_output.toPlainText()
    finally:
        close(w)


def test_a_stack_with_no_axes_is_refused_not_guessed(qapp, ws, monkeypatch):
    """A1-10: no resolved axis order -> the stack is skipped, named, never
    run as "CZYX"."""
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        seen = []
        monkeypatch.setattr(c, "_launch_prediction_worker", seen.extend)
        tick(w, qapp, "P1.tif")
        for r in c._rows:
            r["axes"] = None
        monkeypatch.setattr(c, "_refresh_stacks", lambda *a, **k: None)
        w.predict_btn.setEnabled(True)
        c._run_prediction()
        pump(qapp)
        assert seen == []
        assert T.LOG_NO_AXES in w.log_output.toPlainText()
    finally:
        close(w)


def test_the_step3_preset_fills_the_channels_not_the_model(qapp, ws):
    """V2.6 : Start from "Step 3" fills the two channels from the
    manifest's Cellpose codes and never touches the model card."""
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        card = w.model_active_indicator_label.text()
        snap = c._model_snapshot()
        c._step3_preset = {"run": "default", "recon_config": {},
                           "channels": [2, 1]}
        c._apply_start_from("step3", save=False)
        pump(qapp)
        assert w.chosen_channels() == (1, 0)
        assert w.model_active_indicator_label.text() == card
        assert c._model_snapshot() == snap
    finally:
        close(w)


# ------------------------------------------------------------ re-audit N1-N6


def _file_models(w, qapp, root):
    """A Step 2 lateral + axial model pair on disk (small files)."""
    c = w.controller
    (root / "m").mkdir(exist_ok=True)
    for bucket in ("lateral", "axial"):
        (root / "m" / bucket).write_bytes(b"weights-" + bucket.encode())
        c._on_model_use({"_kind": "step2_run", "bucket": bucket,
                         "weights_path": str(root / "m" / bucket),
                         "human_label": f"run · round 1 ({bucket})"})
    w.set_channel_choice(w.chan1_combo, 0)
    w.set_channel_choice(w.chan2_combo, None)
    pump(qapp)
    return c


def _views():
    return {v: (np.zeros((2, 6, 24, 24), np.float32),
                np.zeros((6, 24, 24), np.float32), None)
            for v in ("xy", "xz", "yz")}


def test_the_finished_snapshot_reaches_the_cache_meta(qapp, ws, monkeypatch):
    """N1: the worker finishes the snapshot on ITS deep copy; the cache
    meta (and so made_with and the manifest) must carry that copy, with
    the sha1s and the versions."""
    import hashlib

    from napari_epicycle.segmentation.controller import _launch_copies
    w = panel(qapp, ws)
    try:
        c = _file_models(w, qapp, ws)
        seen = []
        monkeypatch.setattr(c, "_launch_prediction_worker", seen.extend)
        tick(w, qapp, "P1.tif")
        w.predict_btn.click()
        pump(qapp)
        cfg = _launch_copies(seen)[0]
        M.finish_run_snapshot(cfg["model_snapshot"], {})    # the worker
        got = {}
        monkeypatch.setattr(c, "_persist_prediction_cache",
                            lambda views, name, meta: got.update(meta))
        c._on_prediction_completed("P1.tif", {
            "views": _views(), "source_stack_path": cfg["source_stack_path"],
            "model_snapshot": cfg["model_snapshot"],
            "cache_meta": cfg["cache_meta"]})
        model = got["model"]
        assert set(model["versions"]) == {"cellpose", "torch"}
        assert model["model_fingerprints"]["lateral"]["sha1"] == \
            hashlib.sha1(b"weights-lateral").hexdigest()
        rec = c.state_manager.get_record("P1.tif")
        assert rec["predicted_with"]["versions"] == model["versions"]
    finally:
        close(w)


def test_the_rescale_rule_follows_cellpose():
    """N2: Cellpose 3.1.1.2 ignores rescale whenever diameter > 0
    (models.py:487-490) and uses diam_mean / diameter."""
    assert M.rescale_rule(True, 16.0) == M.RESCALE_FROM_WIDTH
    assert M.rescale_rule(None, 16.0) == M.RESCALE_FROM_WIDTH
    assert M.rescale_rule(True, None) == M.RESCALE_NONE
    assert M.rescale_rule(None, None) == M.RESCALE_TRAINED
    assert M.real_rescale_factor(M.RESCALE_FROM_WIDTH, 15.0, (30.0, 30.0)) \
        == 2.0
    assert M.real_rescale_factor(M.RESCALE_TRAINED, None, (30.0, 20.0)) == 1.5
    assert M.real_rescale_factor(M.RESCALE_NONE, None, None) == 1.0
    assert M.real_rescale_factor(M.RESCALE_TRAINED, None, None) is None
    assert M.model_diams("cyto3", "pretrained") == (30.0, 30.0)
    assert M.model_diams("nuclei", "pretrained") == (17.0, 17.0)


def test_a_cell_width_is_recorded_as_the_rule(qapp, ws, monkeypatch):
    """N2: with a cell width set, the snapshot says the factor comes from
    it, with the pixel width, and the worker computes it from diam_mean."""
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)                       # built-in cyto3
        seen = []
        monkeypatch.setattr(c, "_launch_prediction_worker", seen.extend)
        tick(w, qapp, "P1.tif")
        w.cell_width_um_sb.setValue(7.5)          # 15 px at 0.5 um
        pump(qapp)
        w.predict_btn.click()
        pump(qapp)
        seg = seen[0]["model_snapshot"]["seg"]
        assert seg["rescale_rule"] == M.RESCALE_FROM_WIDTH
        assert seg["diameter_px"] == 15.0
        M.finish_run_snapshot(seen[0]["model_snapshot"], {})
        assert seg["rescale_factor_by_model"]["lateral"] == 2.0   # 30 / 15
        assert seg["rescale_factor"] == 2.0
    finally:
        close(w)


def test_a_stack_with_no_axes_cannot_run():
    """N3: empty axes -> can_run False and a row warning, so the tick, the
    skipped count, the gate and the done count agree."""
    from napari_epicycle.segmentation import _step4_rows as R
    row = R.build_row({"name": "x.tif", "path": "x.tif"},
                      {"stack_shape": [2, 6, 24, 24], "n_channels": 2})
    assert row["can_run"] is False and row["tickable"] is False
    assert T.ROW_WARN_NO_AXES in row["warns_narrow"]


def test_the_idle_line_counts_only_stacks_that_will_run(qapp, tmp_path):
    """N4 / re-audit: "Ready · {n}" counts the ticked stacks that will
    run; a stored tick of an unlisted stack is not counted."""
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"),
                        one_channel=("P2.tif",))
    w = panel(qapp, ws)
    try:
        c = _ready(w, qapp)
        w.set_channel_choice(w.chan2_combo, 1)    # two channels needed
        c._on_channels_changed()
        c.state_manager.set_ticked(["P1.tif", "P2.tif", "gone.tif"])
        c._refresh_stacks()
        pump(qapp)
        assert set(c._stored_ticked()) == {"P1.tif", "P2.tif"}
        text = w.run_status_label.toolTip() or w.run_status_label.text()
        assert text.startswith(T.seg_idle_line(1))
    finally:
        close(w)


def test_an_old_record_gains_dtype_and_axes_how(qapp, ws):
    """N5: a record written before v2.6 (no dtype) re-reads its header."""
    w = panel(qapp, ws)
    try:
        c = w.controller
        rec = c.state_manager.get_record("P1.tif")
        c.state_manager.set_stack_status(
            "P1.tif", c.state_manager.get_stack_status("P1.tif"),
            dtype=None, axes_how=None)
        entry = next(e for e in c._sources if e["name"] == "P1.tif")
        c._fill_record(entry, None)
        rec = c.state_manager.get_record("P1.tif")
        assert rec["dtype"] == "uint8"
        assert rec["axes_how"] == "metadata"
    finally:
        close(w)


def test_a_readopted_manifest_keeps_made_with(qapp, ws, monkeypatch):
    """N6: a stack known only from its run manifest gets the manifest's
    made_with."""
    w = panel(qapp, ws)
    try:
        c = w.controller
        made = {"kind": "pretrained", "lateral": "cyto3", "axial": "cyto3"}
        monkeypatch.setattr(c.run_store, "list_manifests",
                            lambda wd: ["m.json"])
        monkeypatch.setattr(c.run_store, "load_manifest", lambda p: {
            "source_stack_path": str(ws / "old" / "Old.tif"),
            "labels_path": str(ws / "old" / "labels.tif"),
            "made_with": made})
        c._refresh_stacks()
        assert c.state_manager.get_record("Old.tif")["made_with"] == made
    finally:
        close(w)
