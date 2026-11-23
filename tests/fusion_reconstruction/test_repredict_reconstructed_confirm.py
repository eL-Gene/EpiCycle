"""A patch re-predicted from curating (status back to predicted) still has
a 3D reconstruction on disk: the next Reconstruct must show the destructive
dialog for it, whatever the status says."""
from __future__ import annotations

import pytest

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
    _active_run_path_id,
)


class _SM:
    def __init__(self, statuses):
        self._s = statuses

    def get_all_patch_statuses(self):
        return dict(self._s)

    def get_active_run_id(self):
        return "default"


def _ctrl(tmp_path, status, with_manifest):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c.loaded_stack_dir = str(tmp_path / "stacks")
    c.state_manager = _SM({"P.tif": status})
    c._curation_busy = False
    c._patch_load_in_flight = False
    c._ticked_patch_names = lambda: ["P.tif"]
    c._pr_check_files = lambda names: None
    c._pr_runnable = lambda names: (list(names), [], [])
    c.logged = []
    c._append_log = c.logged.append
    c.asked = []
    c._show_rereconstruct_dialog = lambda names: c.asked.append(names) or False
    if with_manifest:
        p = _paths.patch_run_manifest_path(
            c.working_dir, "P.tif", run_id=_active_run_path_id(c))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    return c


def test_a_repredicted_patch_with_a_run_folder_gets_the_destructive_dialog(
        tmp_path):
    c = _ctrl(tmp_path, "predicted", with_manifest=True)
    c._reconstruct_selected()
    assert c.asked == [["P.tif"]]


def test_a_predicted_patch_without_a_reconstruction_does_not(tmp_path):
    c = _ctrl(tmp_path, "predicted", with_manifest=False)
    with pytest.raises(AttributeError):      # runs on past the dialog gate
        c._reconstruct_selected()
    assert c.asked == []


def test_the_predict_again_sentence_says_what_reconstruct_will_replace():
    body = _pr_text.S3_REPREDICT_BODY
    assert "3D labels" in body and "curation" in body
