"""F21 (G1-20): Step 1's Trim / Delete guard names
Step 4 when Step 4 has predicted or reconstructed the patch.

``_downstream._step4_uses`` reads ``step4_state.json`` at the workspace
root, else the older ``artifacts/step4/step4_state.json``, then (after a
corrupt file) the backup, then the legacy ``stack_segmentation_state.json``; a record is the patch's when
its file name's stem, or its source file's stem, IS the patch name.
"""
from __future__ import annotations

import json
import os

import pytest

from napari_epicycle.data_preparation._downstream import (
    DownstreamUse,
    downstream_used_names,
    downstream_uses,
)


def _ws(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return str(ws)


def _state(ws, stacks, legacy=False):
    if legacy:
        path = os.path.join(ws, "stack_segmentation_state.json")
    else:
        path = os.path.join(ws, "artifacts", "step4", "step4_state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 2, "stacks": stacks}, f)


def _src(ws, name):
    return os.path.join(ws, "artifacts", "step1", "patches", name)


def test_a_predicted_patch_names_step4(tmp_path):
    ws = _ws(tmp_path)
    _state(ws, {"P1.tif": {"status": "predicted",
                           "source_stack_path": _src(ws, "P1.tif")}})
    use = downstream_uses(ws, "P1")
    assert use.any
    assert use.step4_kinds == ("prediction",)
    assert use.summary() == "a Step 4 prediction"


def test_a_reconstructed_patch_names_step4s_3d_result(tmp_path):
    ws = _ws(tmp_path)
    _state(ws, {"P1.tif": {"status": "finalized",
                           "labels_path": "x/labels_3d.tif"},
                "P1_copy.tif": {"status": "predicted",
                                "source_stack_path": _src(ws, "P1.tif")}})
    use = downstream_uses(ws, "P1")
    assert use.step4_kinds == ("prediction", "reconstruction")
    assert use.summary() == "Step 4 prediction and reconstruction"


@pytest.mark.parametrize("status", ["unprocessed", "failed"])
def test_nothing_made_means_nothing_named(tmp_path, status):
    ws = _ws(tmp_path)
    _state(ws, {"P1.tif": {"status": status}})
    assert downstream_uses(ws, "P1").any is False


def test_p1_never_claims_p10(tmp_path):
    ws = _ws(tmp_path)
    _state(ws, {"P10.tif": {"status": "reconstructed",
                            "source_stack_path": _src(ws, "P10.tif")}})
    assert downstream_uses(ws, "P1").any is False
    assert downstream_uses(ws, "P10").step4_kinds == ("reconstruction",)


def test_the_legacy_state_file_is_read(tmp_path):
    ws = _ws(tmp_path)
    _state(ws, {"P1.tif": {"status": "reconstructed"}}, legacy=True)
    assert downstream_uses(ws, "P1").step4_kinds == ("reconstruction",)


def test_a_corrupt_state_file_answers_nothing(tmp_path):
    ws = _ws(tmp_path)
    path = os.path.join(ws, "artifacts", "step4", "step4_state.json")
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert downstream_uses(ws, "P1") == DownstreamUse.empty()


def test_the_three_steps_join_in_one_sentence():
    use = DownstreamUse(step2_pairs=2, step2_runs=("cells",),
                        step3_kinds=("reconstruction",),
                        step4_kinds=("reconstruction",))
    assert use.summary() == (
        "2 slices you already curated in Step 2 (run 'cells'), a Step 3 "
        "reconstruction and a Step 4 reconstruction")


def test_the_names_sweep_sees_step4(tmp_path):
    ws = _ws(tmp_path)
    with open(os.path.join(ws, "step1_state.json"), "w",
              encoding="utf-8") as f:
        json.dump({"schema_version": 2, "sources": [],
                   "patches": [{"patch_name": "P1",
                                "relative_path": "artifacts/step1/patches/"
                                                 "P1.tif"}]}, f)
    _state(ws, {"P1.tif": {"status": "predicted"}})
    assert downstream_used_names(ws) == {"P1"}


class _FakeBox:
    last = None
    Warning = 2
    AcceptRole = 0
    RejectRole = 1

    def __init__(self, *_a):
        _FakeBox.last = self
        self.buttons = []

    def setIcon(self, *_a):
        pass

    def setWindowTitle(self, *_a):
        pass

    def setText(self, t):
        self.text = t

    def setInformativeText(self, t):
        self.informative = t

    def addButton(self, label, role):
        self.buttons.append((label, role))
        return (label, role)

    def setDefaultButton(self, b):
        self.default = b

    def exec_(self):
        self._clicked = self.default

    def clickedButton(self):
        return self._clicked


@pytest.mark.parametrize("action,needle", [
    ("trim", "Steps 2, 3 or 4"),
    ("delete", "Its Step 4 results stay on disk under artifacts/step4"),
])
def test_the_confirm_names_step4(monkeypatch, action, needle):
    import types

    qtw = pytest.importorskip("qtpy.QtWidgets")
    monkeypatch.setattr(qtw, "QMessageBox", _FakeBox, raising=False)
    from napari_epicycle.data_preparation.controller import (
        DataPreparationController as C,
    )
    c = types.SimpleNamespace(widget=None)
    use = DownstreamUse(step4_kinds=("prediction",))
    assert C._confirm_downstream(c, "P1", use, action) is False
    box = _FakeBox.last
    assert "P1 is used by a Step 4 prediction" in box.informative
    assert needle in box.informative
    assert "clear or reset" not in box.informative      # R1-11


def test_the_confirm_without_step4_is_unchanged(monkeypatch):
    import types

    qtw = pytest.importorskip("qtpy.QtWidgets")
    monkeypatch.setattr(qtw, "QMessageBox", _FakeBox, raising=False)
    from napari_epicycle.data_preparation.controller import (
        DataPreparationController as C,
    )
    c = types.SimpleNamespace(widget=None)
    use = DownstreamUse(step3_kinds=("reconstruction",))
    C._confirm_downstream(c, "P1", use, "trim")
    assert "anything in Steps 2 or 3 —" in _FakeBox.last.informative


# ------------------------------------------------ K8 (the K-lens audit)


def _step1(ws, *names):
    with open(os.path.join(ws, "step1_state.json"), "w",
              encoding="utf-8") as f:
        json.dump({"schema_version": 2, "sources": [],
                   "patches": [{"patch_name": os.path.splitext(n)[0],
                                "relative_path":
                                    "artifacts/step1/patches/" + n}
                               for n in names]}, f)


def test_an_add_folder_stack_with_the_same_stem_is_not_the_patch(tmp_path):
    """K8a: the record's source path is compared with the patch's own file;
    a stack added from another folder with the same stem is not it."""
    ws = _ws(tmp_path)
    _step1(ws, "P1.tif")
    other = os.path.join(str(tmp_path), "elsewhere", "P1.tif")
    _state(ws, {"P1.tif": {"status": "reconstructed",
                           "source_stack_path": other}})
    assert downstream_uses(ws, "P1").any is False


def test_the_patch_itself_is_matched_by_its_path(tmp_path):
    ws = _ws(tmp_path)
    _step1(ws, "P1.tif")
    _state(ws, {"renamed.tif": {"status": "predicted",
                                "source_stack_path": _src(ws, "P1.tif")}})
    assert downstream_uses(ws, "P1").step4_kinds == ("prediction",)


def test_the_backup_is_read_before_the_legacy_file(tmp_path):
    """K8b / re-audit M2-LOW: Step 4's own order -- a CORRUPT main file
    falls back to the .bak, then the legacy file."""
    ws = _ws(tmp_path)
    main = os.path.join(ws, "artifacts", "step4", "step4_state.json")
    os.makedirs(os.path.dirname(main), exist_ok=True)
    with open(main, "w", encoding="utf-8") as f:
        f.write("{torn")
    bak = os.path.join(ws, "artifacts", "step4", "state_backup",
                       "step4_state.json.bak")
    os.makedirs(os.path.dirname(bak))
    with open(bak, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 2,
                   "stacks": {"P1.tif": {"status": "predicted"}}}, f)
    _state(ws, {"P1.tif": {"status": "reconstructed"}}, legacy=True)
    assert downstream_uses(ws, "P1").step4_kinds == ("prediction",)


def test_curated_is_not_a_step4_status():
    """K8c."""
    from napari_epicycle.data_preparation import _downstream as D
    assert "curated" not in D._STEP4_3D


def test_a_missing_main_file_goes_straight_to_the_legacy_file(tmp_path):
    """Re-audit M2-LOW: with no main file Step 4 reads the legacy file,
    never an orphaned .bak."""
    ws = _ws(tmp_path)
    bak = os.path.join(ws, "artifacts", "step4", "state_backup",
                       "step4_state.json.bak")
    os.makedirs(os.path.dirname(bak))
    with open(bak, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 2,
                   "stacks": {"P1.tif": {"status": "predicted"}}}, f)
    _state(ws, {"P1.tif": {"status": "reconstructed"}}, legacy=True)
    assert downstream_uses(ws, "P1").step4_kinds == ("reconstruction",)


def test_a_copied_workspace_still_names_step4(tmp_path):
    """Re-audit M2: the dev-PC copy flow -- the record's source path is
    absolute to the OLD location; its workspace-relative tail is still the
    patch's own file, so Step 4 is named."""
    ws = _ws(tmp_path)
    _step1(ws, "P1.tif")
    old = "D:/old_pc/project/artifacts/step1/patches/P1.tif"
    _state(ws, {"P1.tif": {"status": "reconstructed",
                           "source_stack_path": old}})
    assert downstream_uses(ws, "P1").step4_kinds == ("reconstruction",)
