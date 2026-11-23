"""When the unsaved-edits question is answered Save and the mask save does
not go through, the guard refuses and says why."""
from __future__ import annotations

from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction import controller as ctl_mod
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)


def test_a_failed_mask_save_in_the_guard_says_so(monkeypatch):
    c = C.__new__(C)
    c.ui = object()
    c._save_in_flight = False
    c._tiers_dirty = False
    c._curation_dirty = True
    c._qc_view_patch_name = "P1.tif"
    c.logged = []
    c._append_log = c.logged.append
    c._save_curation = lambda **_k: None     # still dirty afterwards
    monkeypatch.setattr(ctl_mod.QMessageBox, "question",
                        staticmethod(lambda *a, **k: ctl_mod.QMessageBox.Save))
    assert c._guard_unsaved_changes() is False
    assert _pr_text.S3_SAVE_FAILED in " ".join(c.logged)
