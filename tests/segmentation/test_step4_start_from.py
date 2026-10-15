"""Start from: the three presets write real values
into the widgets; a hand edit flips the combo to Custom."""
from __future__ import annotations

import json

import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import close, make_workspace, panel, pump


def _value(w):
    return w.value() if hasattr(w, "value") else w.isChecked()


def test_the_epicycle_table_is_reconstruction_config():
    from packages.epicycle_reconstruction.config import ReconstructionConfig
    rc = ReconstructionConfig()
    P = M.STEP4_PRESET_EPICYCLE
    assert P["skip_postprocess_cb"] == rc.skip_postprocess
    assert P["n_cpu_sb"] == rc.n_cpu
    assert P["ps_smooth_probs_cb"] == rc.pre_smooth_probs
    assert P["ps_sigma_perp_sb"] == rc.pre_smooth_sigma_perp
    assert P["ps_sigma_inplane_sb"] == rc.pre_smooth_sigma_inplane
    assert P["ps_mask_thresh_sb"] == rc.flow_mask_threshold
    assert P["ps_soft_weight_cb"] == rc.flow_soft_weighting
    groups = {"gradient_descent": rc.gradient_descent,
              "combine_cell_probs": rc.combine_cell_probs,
              "postprocess_binary": rc.postprocess_binary,
              "combine_cell_gradients": rc.combine_cell_gradients,
              "connected_component": rc.connected_component,
              "size_filters": rc.size_filters}
    assert M._preset_from_groups(groups) == {
        k: v for k, v in P.items() if k in M.USEG3D_WIDGET_KEYS}


def test_the_useg3d_mapping_reads_parameters_py():
    params = pytest.importorskip("segment3D.parameters")
    agg = params.get_2D_to_3D_aggregation_params()
    post = params.get_postprocess_segmentation_params()
    P = M.useg3d_preset()
    assert P["gd_iters_sb"] == agg["gradient_descent"]["n_iter"]
    assert P["gd_decay_sb"] == agg["gradient_descent"]["gradient_decay"]
    assert P["ccp_ksize_sb"] == agg["combine_cell_probs"]["ksize"]
    assert P["post_small_obj_sb"] == \
        agg["postprocess_binary"]["remove_small_objects"]
    assert P["gd_tile_x_sb"] == agg["gradient_descent"]["tile_shape"][2]
    assert P["sf_min_size_sb"] == post["size_filters"]["min_size"]
    assert P["sf_max_size_sb"] == -1           # None -> "no limit"
    assert P["ccp_prob_thresh_sb"] == -1       # None -> "Auto"
    # every mapped widget exists on the panel
    for widget in M.USEG3D_WIDGET_KEYS:
        assert widget in T.STEP4_ADVANCED_LABELS


def test_the_step3_preset_fills_only_what_the_manifest_records():
    rc = {"skip_postprocess": False, "pre_smooth_probs": True,
          "gradient_descent": {"n_iter": 100, "gradient_decay": 0.01}}
    out = M.step3_recon_preset(rc)
    assert out == {"skip_postprocess_cb": False, "ps_smooth_probs_cb": True,
                   "gd_iters_sb": 100, "gd_decay_sb": 0.01}


def _manifest(ws, recon):
    d = ws / "artifacts/step3/runs/P1.tif"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"recon_config": recon}),
                                     encoding="utf-8")


@pytest.fixture
def ws(tmp_path):
    ws = make_workspace(tmp_path)
    _manifest(ws, {"method": "direct", "skip_postprocess": True,
                   "pre_smooth_probs": True, "pre_smooth_sigma_perp": 1.0,
                   "gradient_descent": {"n_iter": 100, "gradient_decay": 0.001,
                                        "tile_shape": [64, 128, 128]}})
    return ws


def test_the_combo_offers_step3_only_with_a_manifest(qapp, ws, tmp_path):
    w = panel(qapp, ws)
    try:
        items = [w.start_from_combo.itemText(i)
                 for i in range(w.start_from_combo.count())]
        assert items == [T.SEG_PRESET_STEP3.format(run="Default"),
                         T.SEG_PRESET_EPICYCLE, T.SEG_PRESET_USEG3D]
    finally:
        close(w)
    bare = make_workspace(tmp_path / "b")
    w = panel(qapp, bare)
    try:
        assert w.start_from_combo.findData("step3") < 0
    finally:
        close(w)


def test_choosing_a_preset_writes_the_widgets(qapp, ws):
    w = panel(qapp, ws)
    try:
        c = w.controller
        i = w.start_from_combo.findData("useg3d")
        w.start_from_combo.setCurrentIndex(i)
        w.start_from_combo.activated.emit(i)
        pump(qapp)
        for name, val in M.useg3d_preset().items():
            assert _value(getattr(w, name)) == pytest.approx(val), name
        assert w.start_from_combo.currentData() == "useg3d"
        i = w.start_from_combo.findData("step3")
        w.start_from_combo.activated.emit(i)
        pump(qapp)
        assert w.gd_iters_sb.value() == 100
        assert w.gd_tile_y_sb.value() == 128
        assert w.ps_sigma_perp_sb.value() == pytest.approx(1.0)
        assert c.state_manager.get_session_metadata()["start_from"] == "step3"
        i = w.start_from_combo.findData("epicycle")
        w.start_from_combo.activated.emit(i)
        pump(qapp)
        for name, val in M.STEP4_PRESET_EPICYCLE.items():
            assert _value(getattr(w, name)) == pytest.approx(val), name
    finally:
        close(w)


def test_a_hand_edit_flips_to_custom(qapp, ws):
    w = panel(qapp, ws)
    try:
        w.gd_iters_sb.setValue(77)
        pump(qapp)
        assert w.start_from_combo.currentData() == "custom"
        assert w.start_from_combo.currentText() == T.SEG_PRESET_CUSTOM
        i = w.start_from_combo.findData("epicycle")
        w.start_from_combo.activated.emit(i)
        pump(qapp)
        assert w.start_from_combo.findData("custom") < 0
        assert w.gd_iters_sb.value() == 25
    finally:
        close(w)
