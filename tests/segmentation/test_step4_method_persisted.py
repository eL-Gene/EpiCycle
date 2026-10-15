"""R0: Step 4 persists the 2D-to-3D
method as ``"direct"``, not the combo's item text
"Direct".

``_collect_recon_config`` read ``method_combo.currentText()``, so "Direct"
went into the ``ReconstructionConfig``, the run snapshot (``made_with`` /
``predicted_with``'s ``recon``) and the reconstruction manifest.  The 3D
result is unaffected (the engine runs the direct path either way).  A state
file written before the fix holds "Direct": nothing compares the recorded
method (the chips compare the model files, "changed since Predict" the
source file's fingerprint), so an old record reads exactly as a new one.
"""
from __future__ import annotations

import pytest

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_rows as R

from ._step4_panel import close, make_workspace, panel


@pytest.fixture
def w(qapp, tmp_path):
    w = panel(qapp, make_workspace(tmp_path, patches=("P1.tif",)))
    yield w
    close(w)


def test_the_config_carries_direct(w):
    cfg = w.controller._collect_recon_config()
    assert cfg.method == "direct"
    snap = M.run_snapshot({}, recon_config=cfg)
    assert snap["recon"]["method"] == "direct"


def test_an_item_without_data_falls_back_to_its_text_lowercased(w):
    combo = w.method_combo
    combo.setItemData(combo.currentIndex(), None)
    assert w.controller._collect_recon_config().method == "direct"


def _row(method):
    snap = dict(M.model_snapshot("cyto3", "cyto3", kind="pretrained"),
                recon={"method": method})
    rec = {"status": "reconstructed", "axes": "CZYX",
           "stack_shape": [2, 6, 24, 24], "n_channels": 2,
           "spacing": {"zyx": [2.0, 0.5, 0.5], "source": "step1"},
           "labels_path": "x", "made_with": snap, "predicted_with": snap}
    entry = {"name": "s.tif", "path": "", "origin": "step1",
             "conflict": False}
    return R.build_row(entry, rec, active_model=snap)


def test_an_old_direct_record_reads_as_a_new_one():
    old, new = _row("Direct"), _row("direct")
    assert old == new
    assert not old["changed_since_predict"]
    assert not any(warn for _t, warn in old["chips"])
