"""The configs Step 3 builds are the ``390d532f`` build's, except rows 11
and 16.

``_collect_seg_config`` / ``_collect_recon_config`` as they were at
``390d532f`` are frozen below VERBATIM (``test_the_frozen_copy_is_verbatim``
compares them with ``git show`` when git is there).  Both are run on the SAME
rebuilt panel, after random hand edits, and compared field by field:

* row 11 -- ``compute_masks``: the old panel's tick was ON by default, the
  hidden one is OFF;
* row 16 -- ``rescale``: unticked was ``False`` (a crash), now ``None``.

Row 10 (R2b part 2, deliberately named here): the cell width is in µm.
The collector itself is unchanged -- its ``diameter`` is still the number
in the box (Auto -> None) -- but that number is no longer what Cellpose
gets: each patch's Predict replaces it, on a copy, with the patch's own
pixel number (``_step3_manifests.patch_diameter``; the number as pixels
only for a patch with no voxel size).  ``test_the_diameter_is_the_number_
typed_and_replaced_per_patch`` pins both halves.  The EpiCycle preset's
cell width is 0.0 ("Auto"), the same in any unit, so it did not change.

The old method list held the engine word itself ("direct"); the rebuilt
one shows "Direct" with the data "direct", so the frozen copy reads the
old panel's text through a stand-in combo.
"""
from __future__ import annotations

import dataclasses
import os
import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- frozen

def old_collect_seg_config(self):
    from packages.epicycle_training.config import SegmentationConfig
    ui = self.ui
    chan1 = ui.chan1_combo.currentIndex()
    chan2 = ui.chan2_combo.currentIndex()
    diameter = ui.diameter_sb.value()
    return SegmentationConfig(
        channels=[chan1, chan2],
        normalize=ui.norm_cb.isChecked(),
        rescale=ui.rescale_cb.isChecked(),
        cellprob_threshold=ui.cellprob_thresh.value(),
        flow_threshold=ui.flow_thresh.value(),
        diameter=diameter if diameter > 0 else None,
        batch_size=ui.batch_sb.value(),
        compute_masks=ui.compute_masks_cb.isChecked(),
    )


def old_collect_recon_config(self):
    from packages.epicycle_reconstruction.config import ReconstructionConfig
    ui = self.ui
    prob_thresh_val = ui.ccp_prob_thresh_sb.value()
    prob_thresh = None if prob_thresh_val < 0 else prob_thresh_val
    return ReconstructionConfig(
        method=ui.method_combo.currentText(),
        skip_postprocess=ui.skip_postprocess_cb.isChecked(),
        n_cpu=ui.n_cpu_sb.value(),
        pre_smooth_probs=ui.ps_smooth_probs_cb.isChecked(),
        pre_smooth_flows=ui.ps_smooth_flows_cb.isChecked(),
        pre_smooth_sigma_perp=ui.ps_sigma_perp_sb.value(),
        pre_smooth_sigma_inplane=ui.ps_sigma_inplane_sb.value(),
        flow_mask_threshold=ui.ps_mask_thresh_sb.value(),
        flow_soft_weighting=ui.ps_soft_weight_cb.isChecked(),
        combine_cell_gradients={
            "ksize": ui.ccg_ksize_sb.value(),
            "alpha": ui.ccg_alpha_sb.value(),
            "smooth_sigma": ui.ccg_smooth_sigma_sb.value(),
            "post_sigma": ui.ccg_post_sigma_sb.value(),
        },
        combine_cell_probs={
            "ksize": ui.ccp_ksize_sb.value(),
            "alpha": ui.ccp_alpha_sb.value(),
            "smooth_sigma": ui.ccp_smooth_sigma_sb.value(),
            "threshold_level": ui.ccp_threshold_level_sb.value(),
            "threshold_n_levels": ui.ccp_threshold_n_levels_sb.value(),
            "prob_thresh": prob_thresh,
            "min_prob_thresh": ui.ccp_min_prob_thresh_sb.value(),
            "apply_one_d_p_thresh": ui.ccp_apply_one_d_p_cb.isChecked(),
        },
        gradient_descent={
            "gradient_decay": ui.gd_decay_sb.value(),
            "n_iter": ui.gd_iters_sb.value(),
            "momenta": ui.gd_momenta_sb.value(),
            "do_mp": ui.do_mp_cb.isChecked(),
            "renorm_gradient": ui.gd_renorm_cb.isChecked(),
            "tile_shape": [ui.gd_tile_z_sb.value(), ui.gd_tile_y_sb.value(), ui.gd_tile_x_sb.value()],
            "tile_aspect": [1, 2, 2],
            "tile_overlap_ratio": ui.gd_tile_overlap_sb.value(),
        },
        postprocess_binary={
            "binary_closing": ui.post_closing_sb.value(),
            "binary_dilation": ui.post_dilation_sb.value(),
            "binary_fill_holes": ui.post_fill_cb.isChecked(),
            "remove_small_objects": ui.post_small_obj_sb.value(),
            "extra_erode": ui.post_extra_erode_sb.value(),
        },
        flow_consistency={
            "do_flow_remove": ui.do_flow_remove_cb.isChecked(),
            "flow_threshold": ui.flow_threshold_sb.value(),
            "dtform_method": ui.fc_dtform_combo.currentText(),
            "edt_fixed_point_percentile": ui.fc_edt_percentile_sb.value(),
            "power_dist": None if ui.fc_power_dist_sb.value() < 0 else ui.fc_power_dist_sb.value(),
            "smooth_skel_sigma": ui.fc_smooth_skel_sb.value(),
        },
        size_filters={
            "min_size": ui.sf_min_size_sb.value(),
            "max_size": None if ui.sf_max_size_sb.value() < 0 else ui.sf_max_size_sb.value(),
            "max_size_factor": ui.sf_max_size_factor_sb.value(),
            "do_stats_filter": ui.sf_do_stats_cb.isChecked(),
        },
        indirect_method={
            "dtform_method": ui.im_dtform_combo.currentText(),
            "smooth_binary": ui.im_smooth_binary_sb.value(),
            "smooth_skel_sigma": ui.im_smooth_skel_sb.value(),
            "edt_fixed_point_percentile": ui.im_edt_percentile_sb.value(),
            "power_dist": None if ui.im_power_dist_sb.value() < 0 else ui.im_power_dist_sb.value(),
        },
        connected_component={
            "min_area": ui.cc_min_area_sb.value(),
            "thresh_factor": ui.cc_thresh_factor_sb.value(),
            "smooth_sigma": ui.cc_smooth_sigma_sb.value(),
        },
    )


def _body(src: str) -> str:
    """The lines of a method from its ``ui = self.ui`` line to its
    ``return`` statement's end, dedented, imports dropped."""
    lines = src.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "ui = self.ui")
    out = []
    for ln in lines[i:]:
        out.append(ln.strip())
        if ln.rstrip() == "        )" or ln.rstrip() == "    )":
            break
    return "\n".join(x for x in out if x)


def test_the_frozen_copy_is_verbatim():
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH")
    proc = subprocess.run(
        [git, "show", "390d532f:napari_epicycle/fusion_reconstruction/"
         "controller.py"], cwd=str(REPO), capture_output=True)
    if proc.returncode != 0:
        pytest.skip("390d532f is not in this clone")
    old = proc.stdout.decode("utf-8")
    import inspect
    for name, frozen in (("_collect_seg_config", old_collect_seg_config),
                         ("_collect_recon_config", old_collect_recon_config)):
        start = old.index(f"    def {name}(self)")
        end = old.index("\n    def ", start + 10)
        assert _body(old[start:end]) == _body(inspect.getsource(frozen)), name


# ---------------------------------------------------------------- the build

@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def ui(app):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    yield w
    w.deleteLater()


class _OldMethodCombo:
    """The old list held the engine word as its text."""

    def __init__(self, combo):
        self._c = combo

    def currentText(self):
        return self._c.currentData()


class _OldMasksTick:
    """The old panel's "Compute masks" was ON by default."""

    def isChecked(self):
        return True


def _old_view(ui):
    class View:
        def __getattr__(self, name):
            return getattr(ui, name)
    v = View()
    v.__dict__["method_combo"] = _OldMethodCombo(ui.method_combo)
    v.__dict__["compute_masks_cb"] = _OldMasksTick()
    return SimpleNamespace(ui=v)


def _new(ui):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    me = SimpleNamespace(ui=ui)
    return (C._collect_seg_config(me), C._collect_recon_config(me))


def _old(ui):
    me = _old_view(ui)
    return old_collect_seg_config(me), old_collect_recon_config(me)


#: built and hidden (decisions 11 and 13), read by both collectors
HIDDEN_KNOBS = ("compute_masks_cb", "do_flow_remove_cb", "flow_threshold_sb",
                "fc_dtform_combo", "fc_edt_percentile_sb", "fc_power_dist_sb",
                "fc_smooth_skel_sb", "im_dtform_combo", "im_smooth_binary_sb",
                "im_smooth_skel_sb", "im_edt_percentile_sb",
                "im_power_dist_sb")


def _randomise(ui, seed):
    from qtpy.QtWidgets import QAbstractSpinBox, QCheckBox, QDoubleSpinBox

    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    rng = random.Random(seed)
    for name in sorted(P.STEP3_SETTING_KEYS):
        w = getattr(ui, name)      # the hidden thresholds too: both read them
        if isinstance(w, QCheckBox):
            w.setChecked(rng.random() < 0.5)
        elif isinstance(w, QDoubleSpinBox):
            steps = int(round((w.maximum() - w.minimum()) / w.singleStep()))
            w.setValue(w.minimum() + rng.randint(0, max(steps, 1))
                       * w.singleStep())
        elif isinstance(w, QAbstractSpinBox):
            w.setValue(rng.randint(int(w.minimum()), int(w.maximum())))
    ui.chan1_combo.setCurrentIndex(rng.randrange(ui.chan1_combo.count()))
    ui.chan2_combo.setCurrentIndex(rng.randrange(ui.chan2_combo.count()))
    # the hidden knobs both collectors read (audit B LOW-2)
    for name in HIDDEN_KNOBS:
        w = getattr(ui, name)
        if isinstance(w, QCheckBox):
            w.setChecked(rng.random() < 0.5)
        elif isinstance(w, QDoubleSpinBox):
            steps = int(round((w.maximum() - w.minimum()) / w.singleStep()))
            w.setValue(w.minimum() + rng.randint(0, max(steps, 1))
                       * w.singleStep())
        elif hasattr(w, "setCurrentIndex"):
            w.setCurrentIndex(rng.randrange(w.count()))


def _diff(a, b):
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    return {k: (da[k], db[k]) for k in da if da[k] != db[k]}


def test_the_defaults_build_the_same_configs(ui):
    (seg_new, rec_new), (seg_old, rec_old) = _new(ui), _old(ui)
    assert _diff(rec_old, rec_new) == {}
    assert _diff(seg_old, seg_new) == {"compute_masks": (True, False)}


@pytest.mark.parametrize("seed", range(12))
def test_random_edits_build_the_same_configs(ui, seed):
    _randomise(ui, seed)
    (seg_new, rec_new), (seg_old, rec_old) = _new(ui), _old(ui)
    assert _diff(rec_old, rec_new) == {}
    # the old build had "Compute masks" ticked (row 11); the new collector
    # reads the hidden tick, which the randomiser now drives too
    want = {}
    if not ui.compute_masks_cb.isChecked():
        want["compute_masks"] = (True, False)
    if not ui.rescale_cb.isChecked():
        want["rescale"] = (False, None)          # row 16
    assert _diff(seg_old, seg_new) == want


def test_the_randomiser_reaches_the_groups(ui):
    _randomise(ui, 99)
    rec = _new(ui)[1]
    from packages.epicycle_reconstruction.config import ReconstructionConfig
    assert _diff(ReconstructionConfig(), rec)        # something moved
    assert rec.method == "direct"


def test_the_randomiser_reaches_the_hidden_knobs(ui):
    """Audit B LOW-2: the hidden flow-consistency group, the indirect
    method's widgets and "Compute masks" are read by both collectors, so
    the randomiser drives them too (they hold their built values in the
    app, but a collector that dropped one would otherwise go unseen)."""
    from packages.epicycle_reconstruction.config import ReconstructionConfig
    moved = {"flow_consistency": False, "indirect_method": False,
             "compute_masks": False}
    base = ReconstructionConfig()
    for seed in range(12):
        _randomise(ui, seed)
        seg, rec = _new(ui)
        moved["flow_consistency"] |= (rec.flow_consistency
                                      != base.flow_consistency)
        moved["indirect_method"] |= (rec.indirect_method
                                     != base.indirect_method)
        moved["compute_masks"] |= bool(seg.compute_masks)
    assert all(moved.values()), moved


def test_the_diameter_is_the_number_typed_and_replaced_per_patch(ui):
    """Row 10 (R2b): the collected ``diameter`` is the number in the µm box;
    the per-patch pixel number replaces it on a copy."""
    from napari_epicycle.fusion_reconstruction import _step3_manifests as S
    from napari_epicycle.ui_shell._recon_presets import STEP4_PRESET_EPICYCLE
    assert STEP4_PRESET_EPICYCLE["cell_width_um_sb"] == 0.0     # Auto
    ui.diameter_sb.setValue(0)
    assert _new(ui)[0].diameter is None
    ui.diameter_sb.setValue(8.0)
    seg = _new(ui)[0]
    assert seg.diameter == 8.0
    d, how = S.patch_diameter(8.0, {"zyx": [2.0, 0.5, 0.5]})
    patched = S.with_diameter(seg, d)
    assert (patched.diameter, how) == (16.0, S.DIAMETER_FROM_SPACING)
    assert seg.diameter == 8.0                                   # a copy
    assert _diff(seg, patched) == {"diameter": (8.0, 16.0)}
