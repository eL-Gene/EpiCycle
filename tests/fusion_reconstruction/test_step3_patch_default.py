"""Step 3's patch default.

When nothing is persisted, Step 3 falls back to the patches Step 1 extracted
into the workspace.  The delicate part is *how*: the four hard blocks and
every downstream join read the CONTROLLER attribute `loaded_stack_dir`, while
the persisted mirror is the separate `state_manager.loaded_stack_dir`
assignment.  So `_load_stack_dir(..., persist=False)` sets the first and skips
only the second — the blocks pass, and nothing about the default is written
into `step3_state.json`, where it would be indistinguishable from a user
choice on the next load and could never re-resolve.

Also pinned here: the re-entry refresh re-resolves the default (a patch
extracted in Step 1 since the last visit appears without a Browse) and never
promotes it to a persisted choice on the way.

Qt-free: the controller is built with `__new__` and stub collaborators.
"""
from __future__ import annotations

import json

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)
from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)


class _Item:
    def __init__(self, name):
        self._name = name
        self._selected = False

    def text(self):
        return self._name

    def data(self, role):
        return self._name

    def setSelected(self, v):
        self._selected = v

    def isSelected(self):
        return self._selected


class _PatchList:
    def __init__(self, names=()):
        self._items = [_Item(n) for n in names]

    def count(self):
        return len(self._items)

    def item(self, i):
        return self._items[i]

    def clear(self):
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def selectedItems(self):
        return [i for i in self._items if i.isSelected()]

    def setCurrentItem(self, item):
        for i in self._items:
            i.setSelected(i is item)


class _Line:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value


class _Check:
    def __init__(self):
        self._checked = False

    def blockSignals(self, on):
        pass

    def setChecked(self, v):
        self._checked = v

    def isChecked(self):
        return self._checked


class _UI:
    def __init__(self):
        self.set_workdir_path = _Line()
        self.set_patch_directory_path = _Line()
        self.downsample_mode_cb = _Check()
        self.loaded_patch_list = _PatchList()


def _controller(working_dir="", real_load=False):
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.ui = _UI()
    ctrl.working_dir = working_dir
    ctrl.loaded_stack_dir = ""
    ctrl._stack_dir_is_default = False
    ctrl.state_manager = ReconstructionStateManager()
    ctrl._suppress_publish = False
    ctrl._workflow_shell = None
    ctrl.logs = []
    ctrl._append_log = lambda msg: ctrl.logs.append(msg)
    for name in (
        "_reset_workspace_scoped_caches", "_refresh_patch_state_list",
        "_clear_model_widgets", "_refresh_patch_list_colors",
        "_refresh_summary_dashboard", "_populate_refinement_source_combo",
        "_refresh_refinement_patch_list", "_refresh_per_orientation_summary",
        "_refresh_mode_hint", "_refresh_run_picker", "_post_load_legacy_notice",
        "_warn_patch_run_slug_collisions", "_prompt_step3_migration",
        "_publish_workspace_to_shell",
    ):
        setattr(ctrl, name, lambda *a, **k: None)
    ctrl._restore_persisted_model_selection = lambda: False
    ctrl._discover_workspace_models = lambda: None
    ctrl._patch_axes_cached = lambda name: True
    ctrl._detect_and_cache_patch_axes = lambda name: None
    if not real_load:
        def _fake(d, *, persist=True):
            ctrl.loaded_stack_dir = d
            ctrl._stack_dir_is_default = not persist
            if persist:
                ctrl.state_manager.loaded_stack_dir = d
        ctrl._load_stack_dir = _fake
    return ctrl


def _patches(tmp_path, names=("a.tiff", "b.tiff")):
    d = tmp_path / "artifacts" / "step1" / "patches"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_bytes(b"II*\x00")
    return d


class TestTheDefaultResolver:
    def test_found_when_the_folder_holds_a_tiff(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        assert ctrl._step1_default_patch_dir() == str(d)

    def test_missing_folder_gives_nothing(self, tmp_path):
        ctrl = _controller(str(tmp_path))
        assert ctrl._step1_default_patch_dir() == ""

    def test_empty_folder_gives_nothing(self, tmp_path):
        (tmp_path / "artifacts" / "step1" / "patches").mkdir(parents=True)
        ctrl = _controller(str(tmp_path))
        assert ctrl._step1_default_patch_dir() == ""

    def test_no_workspace_root_scan(self, tmp_path):
        (tmp_path / "loose.tiff").write_bytes(b"II*\x00")
        ctrl = _controller(str(tmp_path))
        assert ctrl._step1_default_patch_dir() == ""

    def test_no_workspace_gives_nothing(self):
        assert _controller("")._step1_default_patch_dir() == ""


class TestAdoptAppliesTheDefault:
    def test_default_used_when_nothing_is_persisted(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.loaded_stack_dir == str(d)

    def test_the_default_is_NOT_persisted(self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.state_manager.loaded_stack_dir in ("", None), (
            "a defaulted folder must never look like a user choice on the "
            "next load")

    def test_the_state_file_records_no_stack_dir(self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        ctrl.state_manager.suppress_save = False
        ctrl.state_manager.save()
        raw = json.loads(
            (tmp_path / "step3_state.json").read_text(encoding="utf-8"))
        blob = json.dumps(raw)
        assert "artifacts/step1/patches" not in blob.replace("\\\\", "/")

    def test_a_persisted_choice_still_wins(self, tmp_path):
        _patches(tmp_path)
        mine = tmp_path / "mine"
        mine.mkdir()
        (tmp_path / "step3_state.json").write_text(json.dumps(
            {"schema_version": 3, "runs": {"default": {
                "patches": {}, "loaded_stack_dir": str(mine)}}}),
            encoding="utf-8")
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.loaded_stack_dir == str(mine)

    def test_nothing_happens_when_step1_has_not_extracted(self, tmp_path):
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.loaded_stack_dir == ""

    def test_one_log_line(self, tmp_path, capsys):
        _patches(tmp_path)
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        out = capsys.readouterr().out
        assert out.count("[WORKSPACE] Step 3 patch folder defaulted") == 1, out

    def test_no_log_line_without_a_default(self, tmp_path, capsys):
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert "defaulted" not in capsys.readouterr().out


class TestTheHardBlocksPassOnTheDefault:
    """The four blocks read the CONTROLLER attribute, not the state field."""

    def test_the_controller_attribute_is_truthy(self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert bool(ctrl.loaded_stack_dir) is True
        assert bool(getattr(ctrl.state_manager, "loaded_stack_dir", "")) is False


class TestPersistFlagOnLoadStackDir:
    def test_persist_true_writes_both(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path), real_load=True)
        ctrl._load_stack_dir(str(d))
        assert ctrl.loaded_stack_dir == str(d)
        assert ctrl.state_manager.loaded_stack_dir == str(d)
        assert ctrl._stack_dir_is_default is False

    def test_persist_false_writes_only_the_attribute(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path), real_load=True)
        ctrl._load_stack_dir(str(d), persist=False)
        assert ctrl.loaded_stack_dir == str(d)
        assert ctrl.state_manager.loaded_stack_dir in ("", None)
        assert ctrl._stack_dir_is_default is True

    def test_browse_persists(self, tmp_path, monkeypatch):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path), real_load=True)
        import napari_epicycle.fusion_reconstruction.controller as mod
        monkeypatch.setattr(
            mod.QFileDialog, "getExistingDirectory",
            staticmethod(lambda *a, **k: str(d)))
        ctrl._browse_stack_dir()
        assert ctrl.state_manager.loaded_stack_dir == str(d)
        assert ctrl._stack_dir_is_default is False


class TestReEntryRefresh:
    def test_the_default_is_resolved_on_entry_after_step1_extracts(
            self, tmp_path):
        """Nothing at adopt time; a patch appears; the next entry finds it."""
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.loaded_stack_dir == ""
        d = _patches(tmp_path)
        ctrl._refresh_lists_on_step_enter()
        assert ctrl.loaded_stack_dir == str(d)
        assert ctrl.state_manager.loaded_stack_dir in ("", None)

    def test_a_re_read_of_the_default_still_does_not_persist(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(real_load=True)
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.loaded_stack_dir == str(d)
        (d / "c.tiff").write_bytes(b"II*\x00")          # the folder changed
        ctrl._refresh_lists_on_step_enter()
        assert ctrl.ui.loaded_patch_list.count() == 3
        assert ctrl.state_manager.loaded_stack_dir in ("", None)

    def test_a_browsed_folder_is_re_read_WITH_persistence(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(real_load=True)
        ctrl.working_dir = str(tmp_path)
        ctrl._load_stack_dir(str(d))                    # as Browse does
        (d / "c.tiff").write_bytes(b"II*\x00")
        ctrl._refresh_lists_on_step_enter()
        assert ctrl.state_manager.loaded_stack_dir == str(d)

    def test_no_workspace_is_a_no_op(self, tmp_path):
        ctrl = _controller()
        ctrl._refresh_lists_on_step_enter()
        assert ctrl.loaded_stack_dir == ""


class TestTheStep3Qualifier:
    """Audit note 2 — Step 3 says where its folder came from, like Step 2."""

    def test_shown_when_the_folder_was_defaulted(self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller(real_load=True)
        ctrl.ui.patch_dir_qualifier = _Label()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.ui.patch_dir_qualifier.visible is True

    def test_hidden_after_a_browse(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path), real_load=True)
        ctrl.ui.patch_dir_qualifier = _Label()
        ctrl._load_stack_dir(str(d))
        assert ctrl.ui.patch_dir_qualifier.visible is False

    def test_hidden_when_a_persisted_choice_is_restored(self, tmp_path):
        _patches(tmp_path)
        mine = tmp_path / "mine"
        mine.mkdir()
        (tmp_path / "step3_state.json").write_text(json.dumps(
            {"schema_version": 3, "runs": {"default": {
                "patches": {}, "loaded_stack_dir": str(mine)}}}),
            encoding="utf-8")
        ctrl = _controller(real_load=True)
        ctrl.ui.patch_dir_qualifier = _Label()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.ui.patch_dir_qualifier.visible is False

    def test_a_ui_without_the_label_is_fine(self, tmp_path):
        """Every Qt-free fake in the suite predates the label."""
        _patches(tmp_path)
        ctrl = _controller(real_load=True)
        ctrl.adopt_workspace(str(tmp_path))          # must not raise
        assert ctrl.loaded_stack_dir


class _Label:
    def __init__(self):
        self.visible = None

    def setVisible(self, v):
        self.visible = bool(v)
